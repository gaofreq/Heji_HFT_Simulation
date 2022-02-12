import numpy as np
from numba import jit

from config import direction_, drivenType_
from library.simulator.util import updateBidStack, updateAskStack
	

@jit(nopython=True)
def clip(value, lower, upper):
	ret = value
	if lower > upper:
		return ret
	if ret < lower:
		ret = lower
	elif ret > upper:
		ret = upper

	return ret

@jit(nopython=True)
def sigmoid(x):
	re = 1 / (1 + np.exp(-x))
	return(re)

def createHandleMktData(computeTheoDiffPredGen):
	@jit(nopython=True)
	def initialize_signals(signals, currentMkt, params, MINMOVE):
		signals.numTicks = 0

		signals.bidLen = 1
		signals.askLen = 1
		signals.bidTopPrice = currentMkt.bid
		signals.askTopPrice = currentMkt.ask
		signals.bidBottomPrice = currentMkt.bid
		signals.askBottomPrice = currentMkt.ask

		signals.volatility = params.initVar

		gammaValue = params.theoGamma
		alphaValue = params.theoAlpha
		if currentMkt.ask - currentMkt.bid > MINMOVE + 0.00001:
			gammaValue = params.theoGamma2
			alphaValue = params.theoAlpha2

		theo = (currentMkt.bid*(currentMkt.az**gammaValue) + currentMkt.ask*(currentMkt.bz**gammaValue)) / (currentMkt.bz**gammaValue + currentMkt.az**gammaValue)
		theo = (1-alphaValue)*theo + alphaValue*currentMkt.last_price

		signals.theoLag = theo
		
		signals.ewmaMid = (currentMkt.bid + currentMkt.ask) * 0.5
		signals.ewmaSpread = currentMkt.spread

		signals.buyVolume = np.log(currentMkt.volume+1) - np.log(0.3*currentMkt.bz+0.7*currentMkt.az)
		signals.sellVolume = np.log(currentMkt.volume+1) - np.log(0.7*currentMkt.bz+0.3*currentMkt.az)
		signals.volume1 = currentMkt.volume
		signals.bidSize = currentMkt.bz
		signals.askSize = currentMkt.az
		signals.bzAll = currentMkt.bzAll
		signals.azAll = currentMkt.azAll
		signals.mainThreadInitialized = True


	@jit(nopython=True)
	def handleMktDataGen(currentMkt, currentTime, currentInsRoot,
						 Position, OrdersHelper, currentSignals,
						 context, params, multiplier, MINMOVE):
		# maxUnitOfRisk = params.maxUnitOfRisk
		# # maxQuoteSize = params.maxQuoteSize
		# # last_price = currentMkt.last_price
		# # volume = currentMkt.volume
		# az = currentMkt.az

		if abs(currentMkt.source) < 1.0e-10:

			DIVIDEADJUST = 0.000001
			eps = OrdersHelper.eps
			MINMOVE = OrdersHelper.MINMOVE

			signals = context.signals[0]
			bidStack = context.bidStack
			askStack = context.askStack
			bidStackNoSpoofing = context.bidStackNoSpoofing
			askStackNoSpoofing = context.askStackNoSpoofing


			gammaValue = params.theoGamma
			alphaValue = params.theoAlpha
			if currentMkt.ask - currentMkt.bid > MINMOVE + 0.00001:
				gammaValue = params.theoGamma2
				alphaValue = params.theoAlpha2

			theo = (currentMkt.bid*(currentMkt.az**gammaValue) + currentMkt.ask*(currentMkt.bz**gammaValue)) / (currentMkt.bz**gammaValue + currentMkt.az**gammaValue)
			theo = (1-alphaValue)*theo + alphaValue*currentMkt.last_price

			circularBuff20 = context.circularBuff20

			if OrdersHelper.sessionStartFlag[0]:
				signals.assistInitialized = False
				signals.mainThreadInitialized = False

			if not signals.mainThreadInitialized:
				initialize_signals(signals, currentMkt, params, MINMOVE)

				bidStack[0] = currentMkt.bz
				bidStackNoSpoofing[0] = min(currentMkt.bz, params.spoofingLevel)
				askStack[0] = currentMkt.az
				askStackNoSpoofing[0] = min(currentMkt.az, params.spoofingLevel)

				circularBuff20[0] = theo

				signals.preBid = currentMkt.bid
				signals.preAsk = currentMkt.ask
				signals.preBidSz = currentMkt.bz
				signals.preAskSz = currentMkt.az
				signals.preLastPrice = currentMkt.last_price					

				computeTheoDiffPredGen(params,signals,currentMkt,currentSignals)            

			if signals.mainThreadInitialized:
				buyOrderCount = 0
				sellOrderCount = 0
				buyOrderTotalCount = 0
				sellOrderTotalCount = 0
				topBuyCount  = 0
				topSellCount = 0

				bidQueueCond = False
				askQueueCond = False
				bidEdgeCond = False
				askEdgeCond = False

				bidQueueAfter = -1.0
				askQueueAfter = -1.0                

				theoPred = 0.0
				theoDiff = 0.0

				if params.qFracMethod == 0:
					updateBidStack(bidStack, bidStackNoSpoofing, currentMkt, signals, params, MINMOVE)
					updateAskStack(askStack, askStackNoSpoofing, currentMkt, signals, params, MINMOVE)

				#### compute signals (need to take of adjust ewma!!!)
				signals.numTicks+=1

				## price and spread
				signals.ewmaMid = signals.ewmaMid*(1-1.0/201.0) + 1.0/201.0*0.5*(currentMkt.bid+currentMkt.ask)
				signals.ewmaSpread = currentMkt.spread*(1.0-params.spreadEwmaDecay)+params.spreadEwmaDecay*signals.ewmaSpread

				## volume and quote size
				tmp = (currentMkt.ask * currentMkt.volume - currentMkt.turnover / multiplier) / currentMkt.spread
				tmp = clip(tmp, 0, currentMkt.volume)
				signals.bidVolume = signals.bidVolume*params.volumeEwmaDecay2 + tmp*(1-params.volumeEwmaDecay)
				tmp = (currentMkt.turnover / multiplier - currentMkt.bid * currentMkt.volume) / currentMkt.spread
				tmp = clip(tmp, 0, currentMkt.volume)
				signals.askVolume = signals.askVolume*params.volumeEwmaDecay2 + tmp*(1-params.volumeEwmaDecay)
				signals.volume1 = signals.volume1*params.volumeEwmaDecay + currentMkt.volume*(1-params.volumeEwmaDecay)

				tmp = np.log(currentMkt.volume+1)- np.log(0.3*currentMkt.bz + 0.7*currentMkt.az)
				signals.buyVolume = signals.buyVolume*params.volumeEwmaDecay + tmp*(1-params.volumeEwmaDecay)
				tmp = np.log(currentMkt.volume+1)- np.log(0.3*currentMkt.az + 0.7*currentMkt.bz)
				signals.sellVolume = signals.sellVolume*params.volumeEwmaDecay + tmp*(1-params.volumeEwmaDecay)

				signals.bidSize = signals.bidSize*params.sizeEwmaDecay + currentMkt.bz*(1-params.sizeEwmaDecay)
				signals.askSize = signals.askSize*params.sizeEwmaDecay + currentMkt.az*(1-params.sizeEwmaDecay)

				## volatility
				if signals.numTicks > 20:
					theoDiff = theo - circularBuff20[np.mod(signals.numTicks, 20)]
				else:
					signals.volatility = params.initVar*20.0

				signals.volatility = (1.0 - params.varEwmaDecay) * theoDiff**2 + params.varEwmaDecay * signals.volatility
				volatility = np.sqrt(signals.volatility / 20.0) / MINMOVE
				

				#### ~~~ update bidVolume, askVolume (different from signals.bidVolume, signals.askVolume) ~~~ ####
				if currentMkt.L2 == 1.0:
					if currentMkt.volume > 0.0:
						## compute vwap
						vwap = (currentMkt.AccTurnOver-signals.estimatedPrevAccTunover)/currentMkt.volume/multiplier
						## make sure vwap are reasonable
						vwap_lower_bound = min(signals.preBid, signals.preAsk, currentMkt.bid, currentMkt.ask, currentMkt.last_price)    
						vwap_upper_bound = max(signals.preBid, signals.preAsk, currentMkt.bid, currentMkt.ask, currentMkt.last_price)
						vwap = clip(vwap, vwap_lower_bound, vwap_upper_bound)						
					else:
						vwap = currentMkt.last_price
					## updates
					signals.estimatedPrevAccTunover = currentMkt.AccTurnOver											
				else:
					vwap = currentMkt.last_price
					## updates
					signals.estimatedPrevAccTunover += currentMkt.volume*vwap*multiplier

				prices = np.sort(np.array([signals.preBid, signals.preAsk, currentMkt.bid, currentMkt.ask]))
				lowerPrice = (prices[prices<=vwap]).max() if prices.min()<=vwap else prices.min() 
				higherPrice = (prices[prices>=vwap]).min() if prices.max()>=vwap else prices.max()
				if lowerPrice == higherPrice:
					lowerPrice = prices.min() 
					higherPrice = prices.max()

				bidVolume = (signals.preAsk - vwap) / (signals.preAsk - signals.preBid) * currentMkt.volume
				askVolume = (vwap - signals.preBid) / (signals.preAsk - signals.preBid) * currentMkt.volume
				bidVolume = clip(bidVolume, 0.0, currentMkt.volume)
				askVolume = clip(askVolume, 0.0, currentMkt.volume)						

				## update theoLag 20 Ticks
				signals.theoLag = theo
				circularBuff20[np.mod(signals.numTicks,20)] = theo

				theoDiffPred, auxSpread = computeTheoDiffPredGen(params,signals,currentMkt,currentSignals)

				newBid =  abs(currentMkt.bid - signals.preBid) > 0.0001
				newAsk =  abs(currentMkt.ask - signals.preAsk) > 0.0001
				newBidSz = abs(currentMkt.bz - signals.preBidSz) > 0.0001
				newAskSz = abs(currentMkt.az - signals.preAskSz) > 0.0001

				signals.preBid = currentMkt.bid
				signals.preAsk = currentMkt.ask
				signals.preBidSz = currentMkt.bz
				signals.preAskSz = currentMkt.az
				signals.preLastPrice = currentMkt.last_price
				signals.bzAll = currentMkt.bzAll
				signals.azAll = currentMkt.azAll
				signals.firstAssistFlag = True
				
				#### ~~~ liquidation ~~~ ####
				if (currentTime+params.timeToClose>OrdersHelper.sessionLastTime):
					if (len(OrdersHelper.activeBuyOrdersLocList)==0 and Position[0].shortPosition == 0 and
						Position[0].longPosition == 0 and len(OrdersHelper.activeSellOrdersLocList) == 0):
						return

					sellOrderCount = 0
					buyOrderCount = 0
					sellOrderTotalCount = 0
					buyOrderTotalCount = 0

					ordersArray = OrdersHelper.ordersArray
					for loc in OrdersHelper.activeBuyOrdersLocList:
						order = ordersArray[loc]

						buyOrderTotalCount+=np.maximum(order.initVolume-order.tradedVolume,0)
						if abs(order.limitPrice-currentMkt.ask)<eps:
							buyOrderCount+=np.maximum(order.initVolume-order.tradedVolume,0)
						else:
							cancelOrder(OrdersHelper,order.refId,currentTime, currentMkt.AS_OF_DATE)


					for loc in OrdersHelper.activeSellOrdersLocList:
						order = ordersArray[loc]

						sellOrderTotalCount+=np.maximum(order.initVolume-order.tradedVolume,0)
						if abs(order.limitPrice-currentMkt.bid)<eps:
							sellOrderCount+=np.maximum(order.initVolume-order.tradedVolume,0)
						else:
							cancelOrder(OrdersHelper,order.refId,currentTime, currentMkt.AS_OF_DATE)


					if (Position[0].longPosition<=Position[0].shortPosition#Position[0].longPosition==0 and Position[0].shortPosition>0
						and buyOrderCount<(Position[0].shortPosition)
						and buyOrderTotalCount<(params.maxUnitOfRisk)):
						submitOrder (OrdersHelper, currentTime,direction_.BUY, currentMkt.ask, 1,Position[0].longPosition, Position[0].shortPosition, buyOrderTotalCount, sellOrderTotalCount, currentMkt.AS_OF_DATE,drivenType_.LIQUIDATION)
					elif (Position[0].longPosition>=Position[0].shortPosition#Position[0].longPosition>0 and Position[0].shortPosition==0
						  and sellOrderCount<(Position[0].longPosition)
						  and sellOrderTotalCount<(params.maxUnitOfRisk)):
						submitOrder (OrdersHelper, currentTime,direction_.SELL, currentMkt.bid, 1,Position[0].longPosition, Position[0].shortPosition, buyOrderTotalCount, sellOrderTotalCount, currentMkt.AS_OF_DATE,drivenType_.LIQUIDATION)

					return


				if not np.isnan(theoDiffPred):
					theoPred = theo + params.alpha*theoDiffPred + params.signalScoreB*clip((theoDiffPred)/volatility, -params.signalScoreC, params.signalScoreC)
					signals.prePred = theoPred 
				else:
					return

				singalTakingSignificant =   abs( (theoPred - theo)  / (MINMOVE + 0.0) / volatility) > params.minStandizedSignalForTaking 
				ordersArray = OrdersHelper.ordersArray				
			 

				minimumEdge = max(
					params.minimumEdge + params.minimumEdgeRate * signals.ewmaMid,
					params.spreadScaler * signals.ewmaSpread + params.varScaler*signals.volatility
				)

				highestBid = -1.0
				lowestAsk = 1000000.0

				#### ~~~ update posJudge ~~~ ####

				for loc in OrdersHelper.activeBuyOrdersLocList:
					order = ordersArray[loc]
					# if (order.limitPrice-currentMkt.ask > -eps) and (currentMkt.volume > 0):
					# 	order.posJudge = 0.0
					if (currentMkt.last_price-order.limitPrice < -eps) and (currentMkt.volume > 0):
						order.posJudge = 0.0
					elif (order.limitPrice-currentMkt.bid > eps):
						order.posJudge = 0.0
					elif abs(order.limitPrice-currentMkt.bid) < eps:
						order.posJudge = max(order.posJudge-bidVolume, 0)						
						order.posJudge = min(order.posJudge, currentMkt.bz)
				for loc in OrdersHelper.activeSellOrdersLocList:
					order = ordersArray[loc]
					# if (order.limitPrice-currentMkt.bid < eps) and (currentMkt.volume > 0):
					# 	order.posJudge = 0.0
					if (currentMkt.last_price-order.limitPrice > eps) and (currentMkt.volume > 0):
						order.posJudge = 0.0
					elif (order.limitPrice-currentMkt.ask < -eps):
						order.posJudge = 0.0
					elif abs(order.limitPrice-currentMkt.ask) < eps:
						order.posJudge = max(order.posJudge-askVolume, 0)						
						order.posJudge = min(order.posJudge, currentMkt.az) 



				#### ~~~ canceling orders ~~~~ ####
				##do not forget to update queueFrontJudge for real trade
				for loc in OrdersHelper.activeBuyOrdersLocList:
					order = ordersArray[loc]

					predEdge = theoPred - order.limitPrice
					bidEdgeCond = params.fracCut * predEdge > minimumEdge * params.posPenLowerBoundCancel * (1/(1+abs(theoPred-theo)/volatility/params.Y_Factor5) if params.Y_Factor4==4 else 1)					

					if params.useNewCancel == True:
						if abs(order.limitPrice - currentMkt.bid) < eps:
							if (signals.volume1 - order.posJudge) <= params.probCancel:
								bidEdgeCond = True

					buyOrderTotalCount += np.maximum(order.initVolume-order.tradedVolume,0)
					if abs(order.limitPrice - currentMkt.bid) <  eps:
						topBuyCount += 1
					if not (bidEdgeCond):
						cancelOrder(OrdersHelper,order.refId,currentTime, currentMkt.AS_OF_DATE)
					elif (currentMkt.bid-order.limitPrice)>(params.quoteRange*MINMOVE+eps):
						cancelOrder(OrdersHelper,order.refId,currentTime, currentMkt.AS_OF_DATE)
						signals.moveAwayCount+=1
					elif abs(order.limitPrice - currentMkt.bid) < (params.riskQuoteRange*MINMOVE + eps):
						buyOrderCount += np.maximum(order.initVolume-order.tradedVolume,0)
						
					if highestBid < order.limitPrice:
						highestBid = order.limitPrice

				for loc in OrdersHelper.activeSellOrdersLocList:
					order = ordersArray[loc]

					predEdge = -theoPred + order.limitPrice
					askEdgeCond = params.fracCut * predEdge > minimumEdge * params.posPenLowerBoundCancel * (1/(1+abs(theoPred-theo)/volatility/params.Y_Factor5) if params.Y_Factor4==4 else 1)

					if params.useNewCancel == True:
						if abs(order.limitPrice - currentMkt.ask) < eps:
							if (signals.volume1 - order.posJudge) <= params.probCancel:
								askEdgeCond = True										

					sellOrderTotalCount+=np.maximum(order.initVolume-order.tradedVolume,0)
					if abs(order.limitPrice - currentMkt.ask) <  eps:
						topSellCount += 1
					if not (askEdgeCond):
						cancelOrder(OrdersHelper,order.refId,currentTime, currentMkt.AS_OF_DATE)
					elif (order.limitPrice-currentMkt.ask)>(params.quoteRange*MINMOVE+eps):
						cancelOrder(OrdersHelper,order.refId,currentTime, currentMkt.AS_OF_DATE)
						signals.moveAwayCount+=1

					elif abs(order.limitPrice-currentMkt.ask) < (params.riskQuoteRange*MINMOVE+eps):
						sellOrderCount += np.maximum(order.initVolume - order.tradedVolume, 0)

					if lowestAsk > order.limitPrice:
						lowestAsk = order.limitPrice
										

				askVol2Size = np.log(signals.askVolume + 1.0) / np.log(signals.askSize + 1.0)
				bidVol2Size = np.log(signals.bidVolume + 1.0) / np.log(signals.bidSize + 1.0)

				varFactor4J = params.varBeta4J * np.exp(params.varAlpha4J * volatility) + params.varBase4J
				varFactor4I = params.varBeta4I * np.exp(params.varAlpha4I * volatility) + params.varBase4I
				varFactor4T = params.varBeta4T * np.exp(params.varAlpha4T * volatility) + params.varBase4T

				volFilter4Joining = volatility < params.volBound4Joining
				volFilter4Improving = volatility < params.volBound4Improving
				varFilter4Taking = volatility > params.volBound4Taking

				#### ~~~ submit buy orders ~~~~ ####			  
				if (Position[0].longPosition - Position[0].shortPosition + buyOrderTotalCount) < params.maxUnitOfRisk and topBuyCount < params.maxNumAtBest:
					## compute bidQueueCond
					riskLim = params.maxUnitOfRisk + Position[0].shortPosition - Position[0].longPosition - buyOrderTotalCount
					buyOneSidedBound = np.maximum(params.oneSidedMaxPos - Position[0].longPosition, 0) - buyOrderTotalCount
				   
					quoteSize = params.maxQuoteSize 
					quantityBound = np.minimum(quoteSize, riskLim)
					quantityBound = np.minimum(quantityBound, buyOneSidedBound)

					futurePos = Position[0].longPosition - Position[0].shortPosition + buyOrderCount
					posPenalty = np.sign(futurePos) * abs(futurePos)**params.penaltyPower
					posFactor = max(params.posPenIntercept+params.posPenCoef*posPenalty, params.posPenLowerBound)

					if params.qFracMethod == 0:
						if signals.bidLen > 2:
							bidQueueAfter = bidStack[signals.bidLen-2] - bidStack[signals.bidLen-3]+params.qI
						elif signals.bidLen > 1:
							bidQueueAfter = bidStack[signals.bidLen-2]+params.qI
						else:
							bidQueueAfter  = params.qI
						bidQueueFrac = (currentMkt.bz) / (bidQueueAfter +DIVIDEADJUST)
					elif params.qFracMethod == 1:
						bidQueueFrac = currentMkt.bz / (signals.bidSize + DIVIDEADJUST)

					
					predCond = (theoPred - theo) > (params.signalFloor + params.signalVarScaler*signals.volatility)

					bidQueueCond = (bidQueueFrac < params.fracCut)
					if params.qFacMethod == 0:
						queueFactor =  (-bidQueueFrac*params.queueEdgeAlpha + params.fracCut)
					else:
						queueFactor = params.fracCut/(params.qAlpha+params.qBeta*np.exp(params.qGamma*bidQueueFrac))
			   
					predEdge =  (theoPred - theo) + (theo - currentMkt.bid)
					bidEdgeCond = queueFactor * predEdge > posFactor * varFactor4J * minimumEdge

					improveBuyPrice = min(currentMkt.bid + params.improveTick*MINMOVE, currentMkt.ask)
					
					predEdgeImprove = theoPred - improveBuyPrice
					mktEdgeImproveCond =  ( theoPred > 0.5*(currentMkt.bid + currentMkt.ask) ) or (params.requirePredCrossMid == False)

					theoEdge = predEdgeImprove
					bidEdgeImproveCond = (theoEdge*params.fractCutImprove > varFactor4I * posFactor * minimumEdge) and ((predEdgeImprove + params.SZALLbeta*currentMkt.bzAll) > params.SZALLThreshold)
					bidEdgeTakeCond = (theoEdge*params.fractCutTake > varFactor4T * posFactor * minimumEdge) and (predEdgeImprove > 0)
					volumeFilterImp = currentMkt.volume < params.maxSnapshotVolumeImp
					volumeFilterForTake = currentMkt.volume > params.minSnapshotVolumeTake
					## EOD LN SS hace
					selfImproveOK = highestBid < currentMkt.bid - 0.5 * MINMOVE
					selfTakeOK = improveBuyPrice < lowestAsk

					extracJoinConditionOK = True
					if params.joinOnlyOnOldLevels:
						extracJoinConditionOK = newBid or newBidSz or (currentMkt.volume < 0.0001)

					if predCond:
						## joining
						if bidQueueCond and bidEdgeCond and volFilter4Joining and extracJoinConditionOK:
							submitOrder(OrdersHelper, currentTime,direction_.BUY, currentMkt.bid, int(quantityBound+0.5), Position[0].longPosition, Position[0].shortPosition, buyOrderTotalCount, sellOrderTotalCount, currentMkt.AS_OF_DATE,drivenType_.TARGET)
						## improving
						elif bidEdgeImproveCond and mktEdgeImproveCond and (currentMkt.ask > improveBuyPrice + eps) and volumeFilterImp and volFilter4Improving and selfImproveOK:
							submitOrder (OrdersHelper, currentTime,direction_.BUY, improveBuyPrice, int(quantityBound+0.5), Position[0].longPosition, Position[0].shortPosition, buyOrderTotalCount, sellOrderTotalCount, currentMkt.AS_OF_DATE,drivenType_.TARGET)
						## taking
						elif bidEdgeTakeCond and mktEdgeImproveCond and (currentMkt.ask < improveBuyPrice + eps) and volumeFilterForTake and singalTakingSignificant and varFilter4Taking and selfImproveOK and selfTakeOK:
							submitOrder (OrdersHelper, currentTime,direction_.BUY, improveBuyPrice, int(quantityBound+0.5), Position[0].longPosition, Position[0].shortPosition, buyOrderTotalCount, sellOrderTotalCount, currentMkt.AS_OF_DATE,drivenType_.TARGET)    

				#### ~~~ submit sell orders ~~~ ####
				if (Position[0].shortPosition - Position[0].longPosition + sellOrderTotalCount) < params.maxUnitOfRisk and topSellCount < params.maxNumAtBest:
					riskLim = params.maxUnitOfRisk - Position[0].shortPosition + Position[0].longPosition - sellOrderTotalCount
					sellOneSidedBound = np.maximum(params.oneSidedMaxPos - Position[0].shortPosition, 0) - sellOrderTotalCount

					quoteSize = params.maxQuoteSize 
					quantityBound = np.minimum(quoteSize, riskLim)
					quantityBound = np.minimum(quantityBound, sellOneSidedBound)

					futurePos = Position[0].shortPosition - Position[0].longPosition + sellOrderCount
					posPenalty = np.sign(futurePos) * abs(futurePos)**params.penaltyPower

					## compute askQueueCond
					if params.qFracMethod == 0:
						if signals.askLen > 2:
							askQueueAfter = askStack[signals.askLen-2] - askStack[signals.askLen-3] + params.qI
						elif signals.askLen > 1:
							askQueueAfter = askStack[signals.askLen-2] + params.qI
						else:
							askQueueAfter  = params.qI
						askQueueFrac = (currentMkt.az) / (askQueueAfter +DIVIDEADJUST)
					elif params.qFracMethod == 1:
						askQueueFrac = currentMkt.az / (signals.askSize + DIVIDEADJUST)

					predCond = (theoPred - theo) < -(params.signalFloor + params.signalVarScaler*signals.volatility)

					askQueueCond = (askQueueFrac < params.fracCut)
					if params.qFacMethod == 0:
						queueFactor =  (-askQueueFrac*params.queueEdgeAlpha + params.fracCut)
					else:
						queueFactor = params.fracCut/(params.qAlpha+params.qBeta*np.exp(params.qGamma*askQueueFrac))
					posFactor = max(params.posPenIntercept + params.posPenCoef*posPenalty, params.posPenLowerBound)
					# predEdge = currentMkt.ask - theoPred
			  
					predEdge =  (currentMkt.ask - theo)  - (theoPred - theo)
					askEdgeCond = queueFactor * predEdge > posFactor * varFactor4J * minimumEdge

					improveSellPrice = max(currentMkt.bid, currentMkt.ask - params.improveTick * MINMOVE)
					predEdgeImprove = improveSellPrice - theoPred
					mktEdgeImproveCond =  ( theoPred < 0.5*(currentMkt.bid + currentMkt.ask) ) or (params.requirePredCrossMid == False)

					theoEdge = predEdgeImprove
					askEdgeImproveCond = (theoEdge*params.fractCutImprove > varFactor4I * posFactor*minimumEdge) and ((predEdgeImprove + params.SZALLbeta*currentMkt.azAll) > params.SZALLThreshold)
					askEdgeTakeCond = (theoEdge*params.fractCutTake > varFactor4T * posFactor*minimumEdge) and (predEdgeImprove > 0)
					volumeFilterImp = currentMkt.volume < params.maxSnapshotVolumeImp
					volumeFilterTake = currentMkt.volume > params.minSnapshotVolumeTake
					## EOD LN SS hace
					selfImproveOK = lowestAsk > currentMkt.ask + 0.5 * MINMOVE 
					selfTakeOK = improveSellPrice > highestBid                    

					extracJoinConditionOK = True
					if params.joinOnlyOnOldLevels:
						extracJoinConditionOK = newAsk or newAskSz or (currentMkt.volume < 0.0001)

					if predCond:
						## joining
						if  askQueueCond and askEdgeCond and volFilter4Joining and extracJoinConditionOK:														
							submitOrder(OrdersHelper, currentTime, direction_.SELL, currentMkt.ask, int(quantityBound+0.5), Position[0].longPosition, Position[0].shortPosition, buyOrderTotalCount, sellOrderTotalCount, currentMkt.AS_OF_DATE,drivenType_.TARGET) 
						## improving
						elif askEdgeImproveCond and mktEdgeImproveCond and (currentMkt.bid < improveSellPrice - eps) and volumeFilterImp and volFilter4Improving and selfImproveOK:
							submitOrder (OrdersHelper, currentTime,direction_.SELL, improveSellPrice, int(quantityBound+0.5), Position[0].longPosition, Position[0].shortPosition, buyOrderTotalCount, sellOrderTotalCount, currentMkt.AS_OF_DATE,drivenType_.TARGET)
						## taking
						elif askEdgeTakeCond and mktEdgeImproveCond and (currentMkt.bid > improveSellPrice - eps) and volumeFilterTake and singalTakingSignificant and varFilter4Taking and selfImproveOK and selfTakeOK:
							submitOrder (OrdersHelper, currentTime,direction_.SELL, improveSellPrice, int(quantityBound+0.5), Position[0].longPosition, Position[0].shortPosition, buyOrderTotalCount, sellOrderTotalCount, currentMkt.AS_OF_DATE,drivenType_.TARGET)

	return handleMktDataGen


@jit(nopython=True)
def computeTheoDiffPred(params,signals,currentMkt,currentSignals):
	return (currentSignals,0.0)


handleMktData = createHandleMktData(computeTheoDiffPred)



#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~ function to handle market data of assistant instrument as signals ~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####

@jit(nopython=True)
def handleSignalMktData(currentMkt, currentTime, currentInsRoot,
				  Position,
				  OrdersHelper,
				  currentSignals,
				  context,params):

	if (currentTime+params.timeToClose>OrdersHelper.sessionLastTime):
		return
	
	if params.useExt == True:
		if (currentInsRoot == params.notActivatedAssistInsRoot1)|(currentInsRoot == params.notActivatedAssistInsRoot2)|(currentInsRoot == params.notActivatedAssistInsRoot3)|(currentInsRoot == params.notActivatedAssistInsRoot4)|(currentInsRoot == params.notActivatedAssistInsRoot5):
			return
 
		signals = context.signals[0]
		theo = signals.theoLag
		preBid = signals.bidTopPrice
		preAsk = signals.askTopPrice
		preSignal = signals.prePred - theo
		pred = params.cmeScaler*currentSignals
		# calculate delta signal
		increAssistSignal = pred - preSignal
        
		ordersArray = OrdersHelper.ordersArray
		MINMOVE = OrdersHelper.MINMOVE

		volatility = np.sqrt(signals.volatility / 20.0) / MINMOVE
		volFilter4Improving = volatility < params.volBound4Improving

		buyOrderTotalCount = 0
		sellOrderTotalCount = 0

		minimumEdge = max(
			params.minimumEdge + params.minimumEdgeRate * signals.ewmaMid,
			params.spreadScaler * signals.ewmaSpread+params.varScaler*signals.volatility
		)

		highestBid = -1.0
		lowestAsk = 1000000.0
        
		# using assistReturn to submit orders
		# only for cme assist contracts
		if params.useAssistReturn == True:  
			currentAssistTheo = (currentMkt.ask*currentMkt.bz + currentMkt.bid*currentMkt.az)/(currentMkt.az+currentMkt.bz)   
			if signals.firstAssistFlag == True:
				signals.firstAssistTheoAfterTarget = currentAssistTheo            
				signals.firstAssistFlag = False
				return
			else:
				increAssistTheo = currentAssistTheo - signals.firstAssistTheoAfterTarget
				theo = (1+params.assistReturnBeta*increAssistTheo/signals.firstAssistTheoAfterTarget)*signals.theoLag
                
		for loc in OrdersHelper.activeBuyOrdersLocList:
			order = ordersArray[loc]

			buyOrderTotalCount += np.maximum(order.initVolume-order.tradedVolume,0)            

			bidEdgeCond = params.fracCut * (theo + pred - order.limitPrice) > minimumEdge * params.cancelScalerOnAssist
			if not (bidEdgeCond):
				cancelOrder(OrdersHelper,order.refId,currentTime, currentMkt.AS_OF_DATE)
			
			if highestBid < order.limitPrice:
				highestBid = order.limitPrice
			
			

		for loc in OrdersHelper.activeSellOrdersLocList:
			order = ordersArray[loc]
			
			sellOrderTotalCount += np.maximum(order.initVolume-order.tradedVolume,0)

			askEdgeCond = params.fracCut * (order.limitPrice - theo - pred) > minimumEdge * params.cancelScalerOnAssist
			if not (askEdgeCond):
				cancelOrder(OrdersHelper,order.refId,currentTime, currentMkt.AS_OF_DATE)

			if lowestAsk > order.limitPrice:
				lowestAsk = order.limitPrice
			
		
		spreadFilter = signals.ewmaSpread > params.spreadLowerBound and signals.ewmaSpread < params.spreadUpperBound


		if (Position[0].longPosition - Position[0].shortPosition + buyOrderTotalCount) < params.maxUnitOfRisk :
			
			riskLim = params.maxUnitOfRisk + Position[0].shortPosition - Position[0].longPosition - buyOrderTotalCount                   
			buyOneSidedBound = np.maximum(params.oneSidedMaxPos - Position[0].longPosition, 0) - buyOrderTotalCount
			quoteSize = params.maxQuoteSize
			quantityBound = np.minimum(quoteSize, riskLim)
			quantityBound = np.minimum(quantityBound, buyOneSidedBound)

			predCond = pred > (params.signalFloor + params.signalVarScaler*signals.volatility)
			predEdgeImprove = theo + pred - (preBid + params.improveTick*MINMOVE)
			bidEdgeImproveCond = (predEdgeImprove > params.minEdge2) and ((predEdgeImprove + params.SZALLbeta*signals.bzAll) > params.SZALLThreshold)
			selfImproveOK = (highestBid < preBid - 0.5 * MINMOVE)
			increAssistTheoOK = (not params.useIncreAssistTheo) | (increAssistTheo>params.increAssistTheoThreshold) # assist theo increment condition 
			increAssistSignalOK = (not params.useIncreAssistSignal) | (increAssistSignal>params.increAssistSignalThreshold) # assist signal increment condition 

			if bidEdgeImproveCond and predCond and selfImproveOK and spreadFilter and volFilter4Improving and increAssistTheoOK and increAssistSignalOK:
				if (preAsk > preBid + 1.5 * MINMOVE):        
					submitOrder (OrdersHelper, currentTime,direction_.BUY, preBid+params.improveTick*MINMOVE, int(quantityBound+0.5),Position[0].longPosition, Position[0].shortPosition, buyOrderTotalCount, sellOrderTotalCount, currentMkt.AS_OF_DATE,drivenType_.ASSIST)
				elif (preAsk < preBid + 1.5 * MINMOVE): 
					submitOrder (OrdersHelper, currentTime,direction_.BUY, preBid+params.improveTick*MINMOVE, int(quantityBound+0.5),Position[0].longPosition, Position[0].shortPosition, buyOrderTotalCount, sellOrderTotalCount, currentMkt.AS_OF_DATE,drivenType_.ASSIST)

		if (Position[0].shortPosition - Position[0].longPosition + sellOrderTotalCount) < params.maxUnitOfRisk:
			
			riskLim = params.maxUnitOfRisk - Position[0].shortPosition + Position[0].longPosition - sellOrderTotalCount
			sellOneSidedBound = np.maximum(params.oneSidedMaxPos - Position[0].shortPosition, 0) - sellOrderTotalCount
			quoteSize = params.maxQuoteSize
			quantityBound = np.minimum(quoteSize, riskLim)
			quantityBound = np.minimum(quantityBound, sellOneSidedBound)

			predCond = pred < -(params.signalFloor + params.signalVarScaler*signals.volatility)            
			predEdgeImprove = preAsk - params.improveTick*MINMOVE - (pred+theo)            
			askEdgeImproveCond = (predEdgeImprove > params.minEdge2) and ((predEdgeImprove + params.SZALLbeta*signals.azAll) > params.SZALLThreshold)
			selfImproveOK = lowestAsk > preAsk + 0.5 * MINMOVE
			increAssistTheoOK = (not params.useIncreAssistTheo) | (increAssistTheo<-params.increAssistTheoThreshold)
			increAssistSignalOK = (not params.useIncreAssistSignal) | (increAssistSignal<-params.increAssistSignalThreshold)  

			if askEdgeImproveCond and predCond and selfImproveOK and spreadFilter and volFilter4Improving and increAssistTheoOK and increAssistSignalOK:
				if (preAsk > preBid + 1.5 * MINMOVE): 
					 submitOrder (OrdersHelper, currentTime,direction_.SELL, preAsk-params.improveTick*MINMOVE, int(quantityBound+0.5),Position[0].longPosition, Position[0].shortPosition, buyOrderTotalCount, sellOrderTotalCount, currentMkt.AS_OF_DATE,drivenType_.ASSIST)
				elif (preAsk < preBid + 1.5 * MINMOVE):
					 submitOrder (OrdersHelper, currentTime,direction_.SELL, preAsk-params.improveTick*MINMOVE, int(quantityBound+0.5),Position[0].longPosition, Position[0].shortPosition, buyOrderTotalCount, sellOrderTotalCount, currentMkt.AS_OF_DATE,drivenType_.ASSIST)

	return 



#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~ function to submit/cancle orders based on trades feedback ~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####

@jit(nopython=True)
def handleTrade(subTrade,
				currentTime,
				Position,
				OrdersHelper,
				context,
				params
				):
	pass