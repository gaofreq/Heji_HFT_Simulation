from numba import jit
from math import ceil
import numpy as np

from config import direction_, of_flag_, orderStatus_, submitType_, logNameType_, drivenType_
from library.simulator.types import Trade,OrderAction,orderLogDtype




@jit(nopython=True, cache=True)
def simulateTrade(OrdersHelper, currentMkt, currentTime, context, simParams):

	#### ~~~ initializations ~~~ ####

	eps = OrdersHelper.eps
	EPS = 10**(-7)
	
	MINMOVE = OrdersHelper.MINMOVE
	TRADEDVOLUMERATIO = OrdersHelper.TRADEDVOLUMERATIO
	originQueueFrac = OrdersHelper.QUEUEFRAC
	QUEUEFRAC = OrdersHelper.QUEUEFRAC
	immediateCancelFailedProb = OrdersHelper.immediateCancelFailedProb
	takingFailedProb = OrdersHelper.takingFailedProb
	improvingFailedProb = OrdersHelper.improvingFailedProb
	joinFailedProb = OrdersHelper.joinFailedProb
	fill1VolumeConstraintFilter = OrdersHelper.fill1VolumeConstraintFilter
	OrdersArray = OrdersHelper.ordersArray
	activeBuyOrdersLocList = OrdersHelper.activeBuyOrdersLocList
	activeSellOrdersLocList = OrdersHelper.activeSellOrdersLocList
	slotsLocList = OrdersHelper.slotsLocList
	toCancelIDList = OrdersHelper.toCancelIDList
	toCancelTimeList = OrdersHelper.toCancelTimeList
	
	## construct condition of non-zero market volume
	mktVolume = int(currentMkt.volume+0.5)
	fill1MktVolumeFilter = True
	if OrdersHelper.fill1MktVolumeFilter:
		fill1MktVolumeFilter = mktVolume > 0
	
	## for calculations and function returns
	totalTradedVolume = 0
	tradeSize = 0
	kBuy = 0      ## an index to be used in activeBuyOrdersLocList
	kSell = 0     ## an index to be used in activeSellOrdersLocList
	trades = []   ## placeholder for final function return


	#### sort active order list by limit price
	insertionSort(a = activeBuyOrdersLocList,  ordersArray = OrdersArray, buyOrSell = direction_.BUY)
	insertionSort(a = activeSellOrdersLocList, ordersArray = OrdersArray, buyOrSell = direction_.SELL)



	## --------------------------------------- ##
	##                   BUY                   ##
	## --------------------------------------- ##	

	#### ~~~ iterate through pending buy orders ~~~ ###
	for loc in activeBuyOrdersLocList:

		oneOrder = OrdersArray[loc]
		id = oneOrder.refId
		matchNow = True      # judge the time diff between order sent time and next tick time
		if oneOrder.drivenType == 1:
			matchNow = (currentTime-oneOrder.sentTime) > simParams.orderLagTime
       
		isOrderLogStatus = oneOrder.isOrderLogStatus
		isOrderLogCancelStatus = oneOrder.isOrderLogCancelStatus		

		## determine whether orders can be canceled immediately
		if (immediateCancelFailedProb < 1.0e-7):
			immediateCancelFailedFlag = False
		elif (1.0-immediateCancelFailedProb) < 1.0e-7:
			immediateCancelFailedFlag = True
		else:
			immediateCancelFailedFlag = (np.random.uniform(0.0,1.0) < immediateCancelFailedProb)

		if (not matchNow):
			immediateCancelFailedFlag = False
		## determine whether orders can be canceled according to orderLog
		if isOrderLogCancelStatus==True:
			orderLogStructArray = OrdersHelper.orderLogStructArray
			tradedVolumeLeft = orderLogStructArray[(orderLogStructArray["LOG_NAME"]==logNameType_.TRADE)&(orderLogStructArray["USED"]==0)&(orderLogStructArray["REF_ID"]==oneOrder.orderLogRefId)]["TRADED_VOLUME"].sum()
			orderLogCancelCond = tradedVolumeLeft == 0
			oneOrder.tradedVolume = max(oneOrder.initVolume - tradedVolumeLeft, oneOrder.tradedVolume)
		else:
			orderLogCancelCond = True

		## handle canceled buy order
		if id in toCancelIDList:
			tempIndex = toCancelIDList.index(id)
			isOrderCancelLag =  (currentTime-toCancelTimeList[tempIndex]) > simParams.orderLagTime
		else:
			isOrderCancelLag = False

		if isOrderCancelLag and not immediateCancelFailedFlag and orderLogCancelCond:
			oneOrder.orderStatus=orderStatus_.CANCELED
			slotsLocList.append(loc)
			oneOrder.firstFlag = False
			continue
		## whether trade at the optimal price
		tradeThroughFlag = False

		#### ~~~ not fully executed ~~~ ####
		if (oneOrder.initVolume - oneOrder.tradedVolume) > 0 and matchNow:

			#### determine QUEUEFRAC by submitType
			if oneOrder.firstFlag:
				#### [<TAKE>: buy limit price == ask price]
				if abs(oneOrder.limitPrice - OrdersHelper.queuePosHelper[0].ask) < eps:
					#### determine QUEUEFRAC
					if takingFailedProb < 1.0e-7:
						QUEUEFRAC = 0.0
					elif (1.0-takingFailedProb) < 1.0e-7:
						QUEUEFRAC = 1.0
					elif np.random.uniform(0.0,1.0)<takingFailedProb:
						QUEUEFRAC = 1.0
					else:
						QUEUEFRAC = originQueueFrac
				#### [<JOIN>: buy limit price == bid price]
				elif abs(oneOrder.limitPrice - OrdersHelper.queuePosHelper[0].bid) < eps:
					#### determine QUEUEFRAC
					if joinFailedProb < 1.0e-7:
						QUEUEFRAC = 0.0
					elif (1.0-joinFailedProb) < 1.0e-7:
						QUEUEFRAC = 1.0
					elif np.random.uniform(0.0,1.0)<joinFailedProb:
						QUEUEFRAC = 1.0
					else:
						QUEUEFRAC = originQueueFrac
				#### [<IMPROVE>: bid price < buy limit price < ask price]
				elif (oneOrder.limitPrice - OrdersHelper.queuePosHelper[0].bid) > eps and (oneOrder.limitPrice - OrdersHelper.queuePosHelper[0].ask) < -eps:
					#### determine QUEUEFRAC
					if improvingFailedProb < 1.0e-7:
						QUEUEFRAC = 0.0
					elif (1.0-improvingFailedProb) < 1.0e-7:
						QUEUEFRAC = 1.0
					elif np.random.uniform(0.0,1.0)<improvingFailedProb:
						QUEUEFRAC = 1.0
					else:
						QUEUEFRAC = originQueueFrac
				#### [<OTHER>]
				else:
					QUEUEFRAC = originQueueFrac

			else:
				QUEUEFRAC = originQueueFrac

			if oneOrder.isTriggered:
				QUEUEFRAC = 0.0		
			#if (oneOrder.drivenType == 1) and ((currentTime-oneOrder.sentTime) > simParams.orderLeadTime):
				#QUEUEFRAC = 0.0

			#### ~~~ <BUY TYPE 5 FILL>: by order log ~~~ ####
			if isOrderLogStatus==True:
				## check with orderLog
				orderLogStructArray = OrdersHelper.orderLogStructArray
				orderLogStructArray = orderLogStructArray[(orderLogStructArray["USED"]==0)&(orderLogStructArray["LOG_NAME"]==logNameType_.TRADE)]
				for orderLog in orderLogStructArray:
					if orderLog["MACHINE_TIMESTAMP"]<=currentTime and orderLog["REF_ID"]==oneOrder.orderLogRefId:
						## determine tradeSize
						tradeSize = oneOrder.initVolume - oneOrder.tradedVolume
						tradeSize = min(tradeSize, int(orderLog["TRADED_VOLUME"]))
						tradeSize = max(tradeSize, 0)
						## update totalTradedVolume
						totalTradedVolume += tradeSize							
						## update order tuple and record trade
						if (tradeSize > 0):
							oneOrder.tradedVolume += tradeSize
							oneTrade = Trade (
								refId = oneOrder.refId,
								of_flag = of_flag_.INIT,
								fillType = 5,
								tradePrice = oneOrder.limitPrice,
								initVolume = oneOrder.initVolume,
								volume = tradeSize,
								direction = direction_.BUY,
								orderSentTime = oneOrder.sentTime,
								executeTime = orderLog["MACHINE_TIMESTAMP"],
								submitType = oneOrder.submitType,
								drivenType = oneOrder.drivenType,
								isDetecting = oneOrder.isDetecting,
								qBeforeTrade = oneOrder.queueFrontFixed,
								AS_OF_DATE = currentMkt.AS_OF_DATE,
								)
							trades.append(oneTrade)
							## update orderLogStructArray
							OrdersHelper.orderLogStructArray[orderLog["INDEX"]]["USED"] = 1
						## update oneOrder
						if orderLog["STATUS"]==0:
							isOrderLogStatus = False
							oneOrder.isOrderLogStatus = False
							oneOrder.queueFrontFixed = 0.0
							oneOrder.queueFrontJudge = 0.0

		   
			#### ~~~ <BUY TYPE 1 FILL>: ask price crosses buy order limit price ~~~ ####
			if( ((oneOrder.limitPrice - currentMkt.ask) > -eps)|((oneOrder.limitPrice<currentMkt.ask-0.1*MINMOVE)&(oneOrder.limitPrice>currentMkt.bid+0.1*MINMOVE)) ) and fill1MktVolumeFilter and isOrderLogStatus==False:
				## determine tradeSize
				tradeSize = oneOrder.initVolume - oneOrder.tradedVolume
				if fill1VolumeConstraintFilter:
					tradeSize = int(TRADEDVOLUMERATIO*mktVolume) if (tradeSize > TRADEDVOLUMERATIO*mktVolume) else tradeSize
					tradeSize = min(tradeSize,mktVolume- totalTradedVolume)
					tradeSize = max(tradeSize,0)
				## update totalTradedVolume
				totalTradedVolume += tradeSize
				## update order tuple and record trade 
				if (tradeSize > 0):
					oneOrder.tradedVolume += tradeSize
					oneTrade = Trade (
						refId = oneOrder.refId,
						of_flag = of_flag_.INIT,
						fillType = 1,
						tradePrice = oneOrder.limitPrice,
						initVolume = oneOrder.initVolume,
						volume = tradeSize,
						direction = direction_.BUY,
						orderSentTime = oneOrder.sentTime,
						executeTime = currentTime,
						submitType = oneOrder.submitType,
						drivenType = oneOrder.drivenType,
						isDetecting = oneOrder.isDetecting,
						qBeforeTrade = oneOrder.queueFrontFixed,
						AS_OF_DATE = currentMkt.AS_OF_DATE,
						)
					trades.append(oneTrade)
				## update tradeThroughFlag
				tradeThroughFlag = True
					
			#### ~~~ <BUY TYPE 2 FILL>: trade through by last price ~~~ ####
			elif ((currentMkt.last_price - oneOrder.limitPrice) < -eps and mktVolume > 0) and isOrderLogStatus==False:
				## determine tradeSize
				tradeSize = oneOrder.initVolume - oneOrder.tradedVolume
				tradeSize = int(TRADEDVOLUMERATIO*mktVolume) if (tradeSize > TRADEDVOLUMERATIO*mktVolume) else tradeSize
				tradeSize = min(tradeSize, mktVolume - totalTradedVolume)
				tradeSize = max(tradeSize, 0)
				## update totalTradedVolume
				totalTradedVolume += tradeSize
				## update order tuple and record trade 
				if (tradeSize > 0):
					oneOrder.tradedVolume += tradeSize
					oneTrade = Trade (
						refId = oneOrder.refId,
						of_flag = of_flag_.INIT,
						fillType = 2,
						tradePrice = oneOrder.limitPrice,
						initVolume = oneOrder.initVolume,
						volume = tradeSize,
						direction = direction_.BUY,
						orderSentTime = oneOrder.sentTime,
						executeTime = currentTime,
						submitType = oneOrder.submitType,
						drivenType = oneOrder.drivenType,
						isDetecting = oneOrder.isDetecting,
						qBeforeTrade = oneOrder.queueFrontFixed,
						AS_OF_DATE = currentMkt.AS_OF_DATE,
						)
					trades.append(oneTrade)
				## update tradeThroughFlag
				tradeThroughFlag = True

				
			## ~~~ <BUY TYPE 3>: vwap cross ~~~ ####
			elif (mktVolume > 0 and (currentMkt.vwap-oneOrder.limitPrice<-EPS)) and not np.isnan(currentMkt.vwap) and isOrderLogStatus==False:
				## determine tradeSize
				tradeSize = oneOrder.initVolume - oneOrder.tradedVolume
				tradeSize = int(TRADEDVOLUMERATIO*mktVolume) if (tradeSize > TRADEDVOLUMERATIO*mktVolume) else tradeSize
				tradeSize = min(tradeSize,mktVolume- totalTradedVolume)
				tradeSize = max(tradeSize,0)
				## update totalTradedVolume
				totalTradedVolume+=tradeSize
				## update order tuple and record trade
				if (tradeSize > 0):
					oneOrder.tradedVolume += tradeSize
					oneTrade = Trade (
						refId = oneOrder.refId,
						of_flag = of_flag_.INIT,
						fillType = 3,
						tradePrice = oneOrder.limitPrice,
						initVolume = oneOrder.initVolume,
						volume = tradeSize,
						direction = direction_.BUY,
						orderSentTime = oneOrder.sentTime,
						executeTime = currentTime,
						submitType = oneOrder.submitType,
						drivenType = oneOrder.drivenType,
						isDetecting = oneOrder.isDetecting,
						qBeforeTrade = oneOrder.queueFrontFixed,
						AS_OF_DATE = currentMkt.AS_OF_DATE,
						)			
					trades.append(oneTrade)
				## update tradeThroughFlag	
				tradeThroughFlag = True
			
			#### ~~~ <BUY TYPE 4>: the case of joining the market ~~~ ####
			elif isOrderLogStatus==False:
				#### queueFrontJudge is nan
				if np.isnan(oneOrder.queueFrontJudge):
					## limit buy price == bid
					if abs(oneOrder.limitPrice - currentMkt.bid) < eps:
						oneOrder.queueFrontJudge = currentMkt.bz
						oneOrder.queueFrontFixed = currentMkt.bz
						QUEUEFRAC = 1.0
					## limit buy price > bid
					elif (oneOrder.limitPrice - currentMkt.bid) > eps:
						oneOrder.queueFrontJudge = 0.0
						oneOrder.queueFrontFixed = 0.0
						QUEUEFRAC = 1.0
				#### queueFrontJudge is not nan
				if not np.isnan(oneOrder.queueFrontJudge):

					#### first time to be filled
					if oneOrder.firstFlag:
						if abs(oneOrder.limitPrice - currentMkt.bid) < eps:
							oneOrder.queueFrontFixed = (1.0-QUEUEFRAC)*oneOrder.queueFrontJudge + QUEUEFRAC*currentMkt.bz                        
						elif (oneOrder.limitPrice - currentMkt.bid) > eps:
							oneOrder.queueFrontFixed = (1.0-QUEUEFRAC)*oneOrder.queueFrontJudge
						else:
							oneOrder.queueFrontFixed = oneOrder.queueFrontJudge

					if mktVolume>0:
						tradeSize = oneOrder.initVolume - oneOrder.tradedVolume
						bidVolume = max(((oneOrder.limitPrice+MINMOVE)-currentMkt.vwap )/MINMOVE *mktVolume,0.0)
						
						## further update bidVolume
						if bidVolume < 0.01:
							bidVolume = 0.0
						else:
							bidVolume = ceil(bidVolume - 0.01)
						if abs(currentMkt.last_price - oneOrder.limitPrice) < eps:
							bidVolume = max(bidVolume, 1.0)

						## determine TMPQUEUEFRAC
						if oneOrder.isTriggered:
							TMPQUEUEFRAC = 1.0
						else:
							TMPQUEUEFRAC = QUEUEFRAC
                            
						#### first time to be filled
						if oneOrder.firstFlag:
							#### can be filled 
							if ((1.0-TMPQUEUEFRAC)*bidVolume - oneOrder.queueFrontFixed) > 0.5:
								## determine tradeSize
								tmpMktVolume = (1.0-TMPQUEUEFRAC)*bidVolume - oneOrder.queueFrontFixed
								tradeSize = int(TRADEDVOLUMERATIO*tmpMktVolume) if (tradeSize > TRADEDVOLUMERATIO*tmpMktVolume) else tradeSize
								tradeSize = min(tradeSize,mktVolume- totalTradedVolume)
								tradeSize = max(tradeSize,0)
								## update totalTradedVolume
								totalTradedVolume += tradeSize
								## update order tuple and record trade
								oneOrder.tradedVolume += tradeSize
								oneOrder.queueFrontFixed = 0.0
								if (tradeSize > 0):
									oneTrade = Trade (
										refId = oneOrder.refId,
										of_flag = of_flag_.INIT,
										fillType = 4,
										tradePrice = oneOrder.limitPrice,
										volume = tradeSize,
										initVolume = oneOrder.initVolume,
										direction = direction_.BUY,
										orderSentTime = oneOrder.sentTime,
										executeTime = currentTime,
										submitType = oneOrder.submitType,
										drivenType = oneOrder.drivenType,
										isDetecting = oneOrder.isDetecting,
										qBeforeTrade = oneOrder.queueFrontFixed,
										AS_OF_DATE = currentMkt.AS_OF_DATE,
										)
									trades.append(oneTrade)
							#### cannot be filled but queueFront decreases
							else:
								oneOrder.queueFrontFixed -= (1.0-TMPQUEUEFRAC)*bidVolume
					  
						#### not first time to be filled
						else:
							#### can be filled
							if (bidVolume-oneOrder.queueFrontFixed)>0.5:
								## determine tradeSize
								tmpMktVolume = bidVolume-oneOrder.queueFrontFixed
								tradeSize = int(TRADEDVOLUMERATIO*tmpMktVolume) if (tradeSize > TRADEDVOLUMERATIO*tmpMktVolume) else tradeSize
								tradeSize = min(tradeSize,mktVolume- totalTradedVolume)
								tradeSize = max(tradeSize,0)
								## update totalTradedVolume
								totalTradedVolume+=tradeSize
								## update order tuple and record trade
								oneOrder.tradedVolume+= tradeSize
								oneOrder.queueFrontFixed = 0.0
								if (tradeSize > 0):
									oneTrade = Trade (
										refId = oneOrder.refId,
										of_flag = of_flag_.INIT,
										fillType = 4,
										tradePrice = oneOrder.limitPrice,
										volume = tradeSize,
										initVolume = oneOrder.initVolume,
										direction = direction_.BUY,
										orderSentTime = oneOrder.sentTime,
										executeTime = currentTime,
										submitType = oneOrder.submitType,
										drivenType = oneOrder.drivenType,
										isDetecting = oneOrder.isDetecting,
										qBeforeTrade = oneOrder.queueFrontFixed,
										AS_OF_DATE = currentMkt.AS_OF_DATE,
										)
									trades.append(oneTrade)
							#### cannot be filled but queueFront decreases
							else:
								oneOrder.queueFrontFixed-=bidVolume
						#### still partially left to be filled
						if (oneOrder.initVolume - oneOrder.tradedVolume) > 0:
							oneOrder.queueFrontJudge -= bidVolume
							oneOrder.queueFrontJudge = np.maximum(oneOrder.queueFrontJudge,0.0)
					#### still partially left to be filled
					if (oneOrder.initVolume - oneOrder.tradedVolume) > 0:
						if abs(currentMkt.bid - oneOrder.limitPrice) < eps:
							oneOrder.queueFrontFixed = np.minimum(oneOrder.queueFrontFixed, currentMkt.bz)
							oneOrder.queueFrontJudge = np.minimum(oneOrder.queueFrontJudge, currentMkt.bz)
						elif (currentMkt.bid - oneOrder.limitPrice) < -eps:
							oneOrder.queueFrontFixed = 0.0
							oneOrder.queueFrontJudge = 0.0
						#elif ((oneOrder.limitPrice - currentMkt.bid) < -0.5*MINMOVE) and ((oneOrder.limitPrice - currentMkt.bid) > -1.5*MINMOVE):
							#oneOrder.queueFrontFixed = np.minimum(oneOrder.queueFrontFixed, currentMkt.bz1)
							#oneOrder.queueFrontJudge = np.minimum(oneOrder.queueFrontJudge, currentMkt.bz1)
							
				
		#### ~~~ update order tuple ~~~ ####
		if abs(oneOrder.initVolume - oneOrder.tradedVolume) > 0 and not tradeThroughFlag:
			## handle canceled buy order
			if id in toCancelIDList and (immediateCancelFailedFlag or (not isOrderCancelLag)):
				oneOrder.orderStatus = orderStatus_.CANCELED
				slotsLocList.append(loc)
				oneOrder.firstFlag = False
			#else:
				#activeBuyOrdersLocList[kBuy] = loc
				#oneOrder.firstFlag = False
				#kBuy += 1
			elif oneOrder.drivenType!=1: 
				activeBuyOrdersLocList[kBuy] = loc
				oneOrder.firstFlag = False 
				kBuy += 1
			else: 
				activeBuyOrdersLocList[kBuy] = loc
				#oneOrder.firstFlag = False 
				kBuy += 1
		else:
			slotsLocList.append(loc)
			oneOrder.orderStatus = orderStatus_.DEAD
			oneOrder.firstFlag = False




	## --------------------------------------- ##
	##                   SELL                  ##
	## --------------------------------------- ##

	#### ~~~ iterate through pending sell orders ~~~ ###
	for loc in activeSellOrdersLocList:

		oneOrder = OrdersArray[loc]
		id = oneOrder.refId
		matchNow = True
		if oneOrder.drivenType == 1:
			matchNow = (currentTime-oneOrder.sentTime) > simParams.orderLagTime
		#isOrderLag = (currentTime-oneOrder.sentTime) > simParams.orderLagTime      # judge the time diff between order sent time and next tick time        
		isOrderLogStatus = oneOrder.isOrderLogStatus
		isOrderLogCancelStatus = oneOrder.isOrderLogCancelStatus		

		## determine whether orders can be canceled immediately
		immediateCancelFailedFlag = False
		if immediateCancelFailedProb < 1.0e-7:
			immediateCancelFailedFlag = False
		elif (1.0 - immediateCancelFailedProb) < 1.0e-7:
			immediateCancelFailedFlag = True
		else:
			immediateCancelFailedFlag = (np.random.uniform(0.0,1.0) < immediateCancelFailedProb)

		## determine whether orders can be canceled according to orderLog
		if isOrderLogCancelStatus==True:
			orderLogStructArray = OrdersHelper.orderLogStructArray
			tradedVolumeLeft = orderLogStructArray[(orderLogStructArray["LOG_NAME"]==logNameType_.TRADE)&(orderLogStructArray["USED"]==0)&(orderLogStructArray["REF_ID"]==oneOrder.orderLogRefId)]["TRADED_VOLUME"].sum()
			orderLogCancelCond = tradedVolumeLeft == 0
			oneOrder.tradedVolume = max(oneOrder.initVolume - tradedVolumeLeft, oneOrder.tradedVolume)
		else:
			orderLogCancelCond = True	

		## handle canceled buy order
		if id in toCancelIDList:
			tempIndex = toCancelIDList.index(id)
			isOrderCancelLag =  (currentTime-toCancelTimeList[tempIndex]) > simParams.orderLagTime
		else:
			isOrderCancelLag = False

		if isOrderCancelLag and not immediateCancelFailedFlag and orderLogCancelCond:
			oneOrder.orderStatus=orderStatus_.CANCELED
			slotsLocList.append(loc)
			oneOrder.firstFlag = False
			continue

		## whether trade at the optimal price
		tradeThroughFlag = False


		#### ~~~ not fully executed ~~~ ####
		if (oneOrder.initVolume-oneOrder.tradedVolume)>0 and matchNow:

			#### determine QUEUEFRAC by submitType
			if oneOrder.firstFlag:
				#### [<TAKE>: sell limit price == bid price]
				if abs(oneOrder.limitPrice - OrdersHelper.queuePosHelper[0].bid) < eps :
					#### determine QUEUEFRAC
					if takingFailedProb < 1.0e-7:
						QUEUEFRAC = 0.0
					elif (1.0-takingFailedProb) < 1.0e-7:
						QUEUEFRAC = 1.0
					elif np.random.uniform(0.0,1.0)<takingFailedProb:
						QUEUEFRAC = 1.0
					else:
						QUEUEFRAC = originQueueFrac
				#### [<JOIN>: sell limit price == ask price]
				elif abs(oneOrder.limitPrice - OrdersHelper.queuePosHelper[0].ask) < eps :
					if joinFailedProb < 1.0e-7:
						QUEUEFRAC = 0.0
					elif (1.0-joinFailedProb) < 1.0e-7:
						QUEUEFRAC = 1.0
					elif np.random.uniform(0.0,1.0)<joinFailedProb:
						QUEUEFRAC = 1.0
					else:
						QUEUEFRAC = originQueueFrac
				#### [<IMPROVE>: bid price < sell limit price < ask price]
				elif (oneOrder.limitPrice- OrdersHelper.queuePosHelper[0].bid) > eps and (oneOrder.limitPrice - OrdersHelper.queuePosHelper[0].ask) < -eps:
					if improvingFailedProb < 1.0e-7:
						QUEUEFRAC = 0.0
					elif (1.0-improvingFailedProb) < 1.0e-7:
						QUEUEFRAC = 1.0
					elif np.random.uniform(0.0,1.0)<improvingFailedProb:
						QUEUEFRAC = 1.0
					else:
						QUEUEFRAC = originQueueFrac
				#### [<OTHER>]
				else:
					QUEUEFRAC = originQueueFrac

			else:
				QUEUEFRAC = originQueueFrac

			if oneOrder.isTriggered:
				QUEUEFRAC = 0.0	
			#if (oneOrder.drivenType == 1) and ((currentTime-oneOrder.sentTime) > simParams.orderLeadTime):
				#QUEUEFRAC = 0.0		   
		   
			#### ~~~ <SELL TYPE 5 FILL>: by order log ~~~ ####
			if isOrderLogStatus==True:
				## check with orderLog
				orderLogStructArray = OrdersHelper.orderLogStructArray
				orderLogStructArray = orderLogStructArray[(orderLogStructArray["USED"]==0)&(orderLogStructArray["LOG_NAME"]==logNameType_.TRADE)]
				for orderLog in orderLogStructArray:
					if orderLog["MACHINE_TIMESTAMP"]<=currentTime and orderLog["REF_ID"]==oneOrder.orderLogRefId:
						## determine tradeSize
						tradeSize = oneOrder.initVolume - oneOrder.tradedVolume
						tradeSize = min(tradeSize, int(orderLog["TRADED_VOLUME"]))
						tradeSize = max(tradeSize, 0)
						## update totalTradedVolume
						totalTradedVolume += tradeSize
						## update order tuple and record trade 
						if (tradeSize > 0):
							oneOrder.tradedVolume += tradeSize
							oneTrade = Trade (
								refId = oneOrder.refId,
								of_flag = of_flag_.INIT,
								fillType = 5,
								tradePrice = oneOrder.limitPrice,
								initVolume = oneOrder.initVolume,
								volume = tradeSize,
								direction = direction_.SELL,
								orderSentTime = oneOrder.sentTime,
								executeTime = orderLog["MACHINE_TIMESTAMP"],
								submitType = oneOrder.submitType,
								drivenType = oneOrder.drivenType,
								isDetecting = oneOrder.isDetecting,
								qBeforeTrade = oneOrder.queueFrontFixed,
								AS_OF_DATE = currentMkt.AS_OF_DATE,
								)
							trades.append(oneTrade)
							## update orderLogStructArray
							OrdersHelper.orderLogStructArray[orderLog["INDEX"]]["USED"] = 1
						## update oneOrder
						if orderLog["STATUS"]==0:
							isOrderLogStatus = False
							oneOrder.isOrderLogStatus = False
							oneOrder.queueFrontFixed = 0.0
							oneOrder.queueFrontJudge = 0.0

					
			#### ~~~ <SELL TYPE 1 FILL>: bid price crosses sell order limit price ~~~ ####
			if( ((oneOrder.limitPrice -currentMkt.bid)<eps) | ((oneOrder.limitPrice<currentMkt.ask-0.1*MINMOVE)&(oneOrder.limitPrice>currentMkt.bid+0.1*MINMOVE)) ) and fill1MktVolumeFilter and isOrderLogStatus==False:
				## determine tradeSize
				tradeSize = oneOrder.initVolume - oneOrder.tradedVolume
				if fill1VolumeConstraintFilter:
					tradeSize = int(TRADEDVOLUMERATIO*mktVolume) if (tradeSize > TRADEDVOLUMERATIO*mktVolume) else tradeSize
					tradeSize = min(tradeSize,mktVolume- totalTradedVolume)
					tradeSize = max(tradeSize,0)
				## update totalTradedVolume
				totalTradedVolume += tradeSize
				## update order tuple and record trade 
				if (tradeSize > 0):
					oneOrder.tradedVolume += tradeSize
					oneTrade = Trade (
						refId = oneOrder.refId,
						of_flag = of_flag_.INIT,
						fillType = 1,
						tradePrice = oneOrder.limitPrice,
						volume = tradeSize,
						initVolume = oneOrder.initVolume,
						direction = direction_.SELL,
						orderSentTime = oneOrder.sentTime,
						executeTime = currentTime,
						submitType = oneOrder.submitType,
						drivenType = oneOrder.drivenType,
						isDetecting = oneOrder.isDetecting,
						qBeforeTrade = oneOrder.queueFrontFixed,
						AS_OF_DATE = currentMkt.AS_OF_DATE,
						)
					trades.append(oneTrade)
				## update tradeThroughFlag
				tradeThroughFlag = True


			#### ~~~ <SELL TYPE 2>: Trade through by last price ~~~ ####
			elif ((currentMkt.last_price - oneOrder.limitPrice) > eps and mktVolume > 0) and isOrderLogStatus==False:
				## determine tradeSize
				tradeSize = oneOrder.initVolume - oneOrder.tradedVolume
				tradeSize = int(TRADEDVOLUMERATIO*mktVolume) if (tradeSize > TRADEDVOLUMERATIO*mktVolume)  else tradeSize
				tradeSize = min(tradeSize,mktVolume- totalTradedVolume)
				tradeSize = max(tradeSize,0)
				## update totalTradedVolume
				totalTradedVolume += tradeSize
				## update order tuple and record trade 
				if (tradeSize > 0):
					oneOrder.tradedVolume += tradeSize
					oneTrade = Trade (
						refId = oneOrder.refId,
						of_flag = of_flag_.INIT,
						fillType = 2,
						tradePrice = oneOrder.limitPrice,
						volume = tradeSize,
						initVolume = oneOrder.initVolume,
						direction = direction_.SELL,
						orderSentTime = oneOrder.sentTime,
						executeTime = currentTime,
						submitType = oneOrder.submitType,
						drivenType = oneOrder.drivenType,
						isDetecting = oneOrder.isDetecting,
						qBeforeTrade = oneOrder.queueFrontFixed,
						AS_OF_DATE = currentMkt.AS_OF_DATE,
						)
					trades.append(oneTrade)
				## update tradeThroughFlag
				tradeThroughFlag = True
				

			#### ~~~ <SELL TYPE 3>: vwap cross ~~~ ####
			elif (mktVolume > 0 and (currentMkt.vwap-oneOrder.limitPrice>EPS)) and not np.isnan(currentMkt.vwap) and isOrderLogStatus==False:
				## determine tradeSize
				tradeSize = oneOrder.initVolume - oneOrder.tradedVolume
				tradeSize = int(TRADEDVOLUMERATIO*mktVolume) if (tradeSize > TRADEDVOLUMERATIO*mktVolume) else tradeSize
				tradeSize = min(tradeSize,mktVolume- totalTradedVolume)
				tradeSize = max(tradeSize,0)
				## update totalTradedVolume
				totalTradedVolume+=tradeSize
				## update order tuple and record trade 
				if (tradeSize > 0):
					oneOrder.tradedVolume+= tradeSize
					oneTrade = Trade (
						refId = oneOrder.refId,
						of_flag = of_flag_.INIT,
						fillType = 3,
						tradePrice = oneOrder.limitPrice,
						initVolume = oneOrder.initVolume,
						volume = tradeSize,
						direction = direction_.SELL,
						orderSentTime = oneOrder.sentTime,
						executeTime = currentTime,
						submitType = oneOrder.submitType,
						drivenType = oneOrder.drivenType,
						isDetecting = oneOrder.isDetecting,
						qBeforeTrade = oneOrder.queueFrontFixed,
						AS_OF_DATE = currentMkt.AS_OF_DATE,
						)
					trades.append(oneTrade)
				## update tradeThroughFlag
				tradeThroughFlag = True


			#### ~~~ <SELL TYPE 4>: the case of joining the market ~~~ ####
			elif isOrderLogStatus==False:
				#### queueFrontJudge is nan
				if np.isnan(oneOrder.queueFrontJudge):
					## limit sell price == ask
					if abs(oneOrder.limitPrice-currentMkt.ask)<eps:
						oneOrder.queueFrontJudge = currentMkt.az
						oneOrder.queueFrontFixed = currentMkt.az
						QUEUEFRAC = 1.0
					## limit buy price < ask
					elif (oneOrder.limitPrice-currentMkt.ask)<-eps:
						oneOrder.queueFrontJudge = 0.0
						oneOrder.queueFrontFixed = 0.0
						QUEUEFRAC = 1.0

				#### queueFrontJudge is not nan
				if not np.isnan(oneOrder.queueFrontJudge):

					#### first time to be filled
					if oneOrder.firstFlag:
						if abs(oneOrder.limitPrice-currentMkt.ask)<eps:
							oneOrder.queueFrontFixed= (1.0-QUEUEFRAC)*oneOrder.queueFrontJudge + QUEUEFRAC*currentMkt.az
						elif (oneOrder.limitPrice-currentMkt.ask)<-eps:
							oneOrder.queueFrontFixed =  (1.0-QUEUEFRAC)*oneOrder.queueFrontJudge
						else:
							oneOrder.queueFrontFixed =  oneOrder.queueFrontJudge
					if mktVolume>0:
						tradeSize = oneOrder.initVolume - oneOrder.tradedVolume
						askVolume = max((-(oneOrder.limitPrice-MINMOVE)+currentMkt.vwap )/MINMOVE *mktVolume,0.0)
						
						## further update bidVolume
						if askVolume<0.01:
							askVolume = 0.0
						else:
							askVolume = ceil(askVolume-0.01)
						if abs(currentMkt.last_price-oneOrder.limitPrice)<eps:
							askVolume = max(askVolume,1.0)

						## determine TMPQUEUEFRAC
						if oneOrder.isTriggered:
							TMPQUEUEFRAC = 1.0
						else:
							TMPQUEUEFRAC = QUEUEFRAC

						#### first time to be filled
						if oneOrder.firstFlag:
							#### can be filled 
							if ((1.0-TMPQUEUEFRAC)*askVolume-oneOrder.queueFrontFixed)>0.5:
								## determine tradeSize
								tmpMktVolume = (1.0-TMPQUEUEFRAC)*askVolume-oneOrder.queueFrontFixed
								tradeSize = int(TRADEDVOLUMERATIO*tmpMktVolume) if (tradeSize > TRADEDVOLUMERATIO*tmpMktVolume)  else tradeSize
								tradeSize = min(tradeSize,mktVolume- totalTradedVolume)
								tradeSize = max(tradeSize,0)
								## update totalTradedVolume
								totalTradedVolume += tradeSize
								## update order tuple and record trade
								oneOrder.tradedVolume += tradeSize
								oneOrder.queueFrontFixed = 0.0
								if (tradeSize > 0):
									oneTrade = Trade (
										refId = oneOrder.refId,
										of_flag = of_flag_.INIT,
										fillType = 4,
										tradePrice = oneOrder.limitPrice,
										initVolume = oneOrder.initVolume,
										volume = tradeSize,
										direction = direction_.SELL,
										orderSentTime = oneOrder.sentTime,
										executeTime = currentTime,
										submitType = oneOrder.submitType,
										drivenType = oneOrder.drivenType,
										isDetecting = oneOrder.isDetecting,
										qBeforeTrade = oneOrder.queueFrontFixed,
										AS_OF_DATE = currentMkt.AS_OF_DATE,
										)
									trades.append(oneTrade)
							#### cannot be filled but queueFront decreases
							else:
								oneOrder.queueFrontFixed-=(1.0-TMPQUEUEFRAC)*askVolume
							 
						#### not first time to be filled
						else:
							#### can be filled
							if (askVolume-oneOrder.queueFrontFixed)>0.5:
								## determine tradeSize
								tmpMktVolume = askVolume-oneOrder.queueFrontFixed
								tradeSize = int(TRADEDVOLUMERATIO*tmpMktVolume) if (tradeSize > TRADEDVOLUMERATIO*tmpMktVolume) else tradeSize
								tradeSize = min(tradeSize,mktVolume- totalTradedVolume)
								tradeSize = max(tradeSize,0)
								## update totalTradedVolume
								totalTradedVolume += tradeSize
								## update order tuple and record trade
								oneOrder.tradedVolume += tradeSize
								oneOrder.queueFrontFixed = 0.0
								if (tradeSize>0):
									oneTrade = Trade (
										refId = oneOrder.refId,
										of_flag = of_flag_.INIT,
										fillType = 4,
										tradePrice = oneOrder.limitPrice,
										initVolume = oneOrder.initVolume,
										volume = tradeSize,
										direction = direction_.SELL,
										orderSentTime = oneOrder.sentTime,
										executeTime = currentTime,
										submitType = oneOrder.submitType,
										drivenType = oneOrder.drivenType,
										isDetecting = oneOrder.isDetecting,
										qBeforeTrade = oneOrder.queueFrontFixed,
										AS_OF_DATE = currentMkt.AS_OF_DATE,
										)
									trades.append(oneTrade)
							#### cannot be filled but queueFront decreases
							else:
								oneOrder.queueFrontFixed -= askVolume
						#### still partially left to be filled
						if (oneOrder.initVolume - oneOrder.tradedVolume) > 0:
							oneOrder.queueFrontJudge -= askVolume
							oneOrder.queueFrontJudge = np.maximum(oneOrder.queueFrontJudge, 0.0) 
					#### still partially left to be filled
					if (oneOrder.initVolume - oneOrder.tradedVolume) > 0:
						if abs(currentMkt.ask - oneOrder.limitPrice) < eps:
							oneOrder.queueFrontFixed = np.minimum(oneOrder.queueFrontFixed,currentMkt.az)
							oneOrder.queueFrontJudge = np.minimum(oneOrder.queueFrontJudge,currentMkt.az)
						elif (currentMkt.ask - oneOrder.limitPrice) > eps:
							oneOrder.queueFrontFixed = 0.0
							oneOrder.queueFrontJudge = 0.0
						#elif (oneOrder.limitPrice - currentMkt.ask > 0.5*MINMOVE) and ((oneOrder.limitPrice - currentMkt.ask) < 1.5*MINMOVE):
							#oneOrder.queueFrontFixed = np.minimum(oneOrder.queueFrontFixed,currentMkt.az1)
							#oneOrder.queueFrontJudge = np.minimum(oneOrder.queueFrontJudge,currentMkt.az1)	

							
		#### ~~~ update order tuple ~~~ ####
		if abs(oneOrder.initVolume - oneOrder.tradedVolume) > 0 and not tradeThroughFlag:
			## handle canceled buy order
			if id in toCancelIDList and (immediateCancelFailedFlag or (not isOrderCancelLag)):
				oneOrder.orderStatus = orderStatus_.CANCELED
				slotsLocList.append(loc)
				oneOrder.firstFlag = False 
			elif oneOrder.drivenType!=1: 
				activeSellOrdersLocList[kSell] = loc
				oneOrder.firstFlag = False 
				kSell += 1
			else: 
				activeSellOrdersLocList[kSell] = loc
				#oneOrder.firstFlag = False 
				kSell += 1
                
		else:
			slotsLocList.append(loc)
			oneOrder.orderStatus = orderStatus_.DEAD
			oneOrder.firstFlag = False


	## ------------------------------------------------ ##
	##    remove the executed from active order list    ##
	## ------------------------------------------------ ##
	for i in range(len(activeBuyOrdersLocList)-kBuy):
		activeBuyOrdersLocList.pop()

	for i in range(len(activeSellOrdersLocList)-kSell):
		activeSellOrdersLocList.pop()

	#### ~~~ function return ~~~ ####
	return trades 




#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~ function to sort orders by limit price ~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####

@jit(nopython=True, cache=True)
def insertionSort(a, ordersArray, buyOrSell):

	for i in range(1,len(a)):
		x = a[i]
		price_x = ordersArray[x].limitPrice
		j = i - 1
		while j >= 0 and (ordersArray[a[j]].limitPrice-price_x)*buyOrSell<-0.0001:
			a[j+1] = a[j]
			j = j - 1

		a[j+1] = x



#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~ function to cancel order in strategy ~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####

@jit(nopython=True, cache=True)
def cancelOrder(OrdersHelper, id, currentTime, AS_OF_DATE):
	#### identify order loc in ordersArray
	loc = np.where(OrdersHelper.ordersArray["refId"]==id)[0][-1]
	ordersArrayCancel = OrdersHelper.ordersArray[loc]

	#### check with orderLog
	if OrdersHelper.useOrderLog:
		orderLogStructArray = OrdersHelper.orderLogStructArray
		orderLogStructArray = orderLogStructArray[(orderLogStructArray["REF_ID"]==ordersArrayCancel.orderLogRefId)]
		for orderLog in orderLogStructArray:
			## prevent cancel before insert
			if orderLog["INSERT_MACHINE_TIMESTAMP"] > currentTime:
				return()
			## match cancel in orderLog			
			if orderLog["MACHINE_TIMESTAMP"]<=currentTime and orderLog["LOG_NAME"]==logNameType_.CANCEL and orderLog["USED"]==0:
				ordersArrayCancel.isOrderLogCancelStatus = True
				OrdersHelper.orderLogStructArray[orderLog["INDEX"]]["USED"] = 1



	## update toCancelIDList
	OrdersHelper.toCancelIDList.append(id)
	OrdersHelper.toCancelTimeList.append(currentTime)

	if OrdersHelper.saveOrders:
		OrdersHelper.orderActionList.append(
			OrderAction(
				refId = id,
				actionTime = currentTime,
				initVolume = -1,
				cancelling= True,
				direction = -1000,
				limitPrice= 0.0,
				detRefId = -1,
				isDetecting = False,
				longPosition = -1000,
				shortPosition = -1000,
				outStandingBuy = -1000,
				outStandingSell = -1000,
				submitType = -1,
				drivenType = -1,
				isOrderLogMatched = 1*ordersArrayCancel.isOrderLogCancelStatus,
				orderLogRefId = -1,
				AS_OF_DATE = AS_OF_DATE,
				)
			)



#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~ function to submit order in strategy ~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####

# the argument refId is a one element list
@jit(nopython=True, cache=True)
def submitOrder(OrdersHelper, 
				currentTime, 
				direction, 
				limitPrice, 
				initVolume,
				longPosition,
				shortPosition,
				outStandingBuy, 
				outStandingSell, 
				AS_OF_DATE,
				drivenType,
				isDetectingTriggered = False, 
				isDetecting = False,
				detRefId = -1,
				precisePosition = -1.0,
				):

	#### need non-zero initVolume
	if initVolume ==0:
		return() 

	#### initializations
	eps = OrdersHelper.eps
	MINMOVE = OrdersHelper.MINMOVE
	queuePosHelper = OrdersHelper.queuePosHelper[0]
	refId = OrdersHelper.refId
	refId[0]+=1
	posJudge = 0.0
	loc = 0

	#### append to active order lists
	if len(OrdersHelper.slotsLocList):
		loc = OrdersHelper.slotsLocList.pop()
		if direction == direction_.BUY:
			OrdersHelper.activeBuyOrdersLocList.append(loc)
		elif direction == direction_.SELL:
			OrdersHelper.activeSellOrdersLocList.append(loc)
	else:
		print("OrdersHelper.ordersArray is Full. Check your maximum positions!")

	#### determine submitType
	submitType = -1
	#### <BUY>
	if direction==direction_.BUY:
		if abs(limitPrice - queuePosHelper.bid) < eps:
			posJudge = queuePosHelper.bz
			submitType = submitType_.JOINING
		elif (limitPrice - queuePosHelper.bid) > eps:
			posJudge = 0.0
			if (limitPrice - queuePosHelper.ask) < -eps:
				submitType = submitType_.IMPROVING
			else:
				submitType = submitType_.TAKING
		elif abs(limitPrice + MINMOVE - queuePosHelper.bid) < eps:
			posJudge = queuePosHelper.bz1
			submitType = submitType_.JOINING
		else:
			posJudge = np.nan
			submitType = submitType_.JOINING
	#### <SELL>
	else:
		if abs(limitPrice - queuePosHelper.ask) < eps:
			posJudge = queuePosHelper.az
			submitType = submitType_.JOINING
		elif (limitPrice - queuePosHelper.ask) < -eps:
			posJudge = 0.0
			if (limitPrice - queuePosHelper.bid) > eps:
				submitType = submitType_.IMPROVING
			else:
				submitType = submitType_.TAKING
		elif abs(limitPrice - MINMOVE - queuePosHelper.ask) < eps:
			posJudge = queuePosHelper.az1
			submitType = submitType_.JOINING
		else:
			posJudge = np.nan
			submitType = submitType_.JOINING

	#### update ordersArrayEntry
	ordersArrayEntry = OrdersHelper.ordersArray[loc]
	ordersArrayEntry.direction = direction
	ordersArrayEntry.refId = refId[0]
	ordersArrayEntry.initVolume = initVolume
	ordersArrayEntry.tradedVolume = 0
	ordersArrayEntry.queueFrontFixed = np.nan
	ordersArrayEntry.posJudge = posJudge
	if abs(precisePosition +1)<1.0e-10:
		ordersArrayEntry.queueFrontJudge =  posJudge
		ordersArrayEntry.isTriggered = False 
	else:
		ordersArrayEntry.queueFrontJudge = precisePosition*1.0
		ordersArrayEntry.isTriggered = True	
	ordersArrayEntry.sentTime = currentTime
	ordersArrayEntry.firstFlag = True
	ordersArrayEntry.deepQuoteFlag = np.isnan(posJudge)
	ordersArrayEntry.orderStatus = orderStatus_.LIVE
	ordersArrayEntry.limitPrice = limitPrice
	ordersArrayEntry.submitType = submitType
	ordersArrayEntry.drivenType = drivenType
	ordersArrayEntry.isDetectingTriggered = isDetectingTriggered
	ordersArrayEntry.isDetecting = isDetecting
	ordersArrayEntry.isOrderLogStatus = False
	ordersArrayEntry.orderLogRefId = -1
	ordersArrayEntry.isOrderLogCancelStatus = False	
	ordersArrayEntry.isOpen = shortPosition==0 if direction==1 else longPosition==0
	#### check with orderLog
	if OrdersHelper.useOrderLog:
		orderLogStructArray = OrdersHelper.orderLogStructArray
		orderLogStructArray = orderLogStructArray[(orderLogStructArray["USED"]==0)&(orderLogStructArray["LOG_NAME"]==logNameType_.SUBMIT)]
		for orderLog in orderLogStructArray:
			if orderLog["MACHINE_TIMESTAMP"]==currentTime and orderLog["DIRECTION"]==direction and abs(orderLog["LIMIT_PRICE"]-limitPrice)<eps:
				ordersArrayEntry.isOrderLogStatus = True
				ordersArrayEntry.orderLogRefId = orderLog["REF_ID"]
				OrdersHelper.orderLogStructArray[orderLog["INDEX"]]["USED"] = 1
	#### save orders
	if OrdersHelper.saveOrders:
		OrdersHelper.orderActionList.append(
			OrderAction(
				refId = refId[0],
				actionTime = currentTime,
				initVolume = initVolume,
				cancelling= False,
				direction = direction,
				limitPrice = limitPrice,
				detRefId = detRefId,
				isDetecting = isDetecting,
				longPosition = longPosition,
				shortPosition = shortPosition,
				outStandingBuy = outStandingBuy,
				outStandingSell = outStandingSell,
				submitType = submitType,
				drivenType = drivenType,
				isOrderLogMatched = 1*ordersArrayEntry.isOrderLogStatus,
				orderLogRefId = ordersArrayEntry.orderLogRefId,
				AS_OF_DATE = AS_OF_DATE,
				)
			)



		