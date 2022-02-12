from numba import jit 
import numpy as np

from library.simulator.types import OrderAction,PositionDtype,OrdersHelper,OrderDtype,QueuePosHelperDtype,TradeResults,Trade
from library.simulator.tradeManager import update,updatePosition
from config import direction_, of_flag_

#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~ function to run simulation (tick-level data) ~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####

@jit(nopython = True)
def simulationForSession(data,
						time,
						insRoot, 
						signals, 
						params, 
						simParams,
						context,
						preSessionProfits, 
						tradeInsRoot,
						multiplier,
						MINMOVE,
						orderLog,
						sessionNum,
						):

	#### ~~~ initilizations ~~~ ####

	dataLen = len (data)
	pos = np.zeros(1,dtype = PositionDtype)
	pos[0].closeProfits = preSessionProfits
	pos[0].closeRate = simParams.closeRate
	pos[0].openRate = simParams.openRate
	pos[0].PERROUNDFEE  = simParams.PERROUNDFEE

	lastIndex = -1

	helper = OrdersHelper(
		activeBuyOrdersLocList = [0],    ## pending buy orders
		activeSellOrdersLocList = [0],   ## pending sell orders
		myTopLevelQuantity = np.zeros(2, dtype=np.int64),
		slotsLocList = list(range(10*params.maxUnitOfRisk)),
		toCancelIDList = [-1],
		toCancelTimeList = [time[0]],  
		refId = [sessionNum*10000+0],
		queuePosHelper = np.empty(1, dtype=QueuePosHelperDtype),
		ordersArray = np.empty(10*params.maxUnitOfRisk, dtype=OrderDtype),
		sessionStartFlag = [True],
		sessionLastTime = time[lastIndex],
		eps = MINMOVE / 10.0,
		MINMOVE = MINMOVE, 
		QUEUEFRAC = simParams.QUEUEFRAC,
		fastTriggerWt = simParams.fastTriggerWt,
		TRADEDVOLUMERATIO = simParams.TRADEDVOLUMERATIO,
		FILLEDVOLUMERATIO = simParams.FILLEDVOLUMERATIO,
		saveOrders = simParams.saveOrders,
		orderActionList = [OrderAction (refId = 0,
										actionTime = time[0],
										initVolume = 0,
										cancelling =True,
										direction = direction_.SELL,
										limitPrice = 0.0,
										detRefId = -1,
										isDetecting = False,
										longPosition = 0,
										shortPosition = 0,
										outStandingBuy = 0,
										outStandingSell = 0,
										submitType = -1,
										drivenType = -1,
										isOrderLogMatched = -1,
										orderLogRefId = -1,
										AS_OF_DATE = 0.0)],
		immediateCancelFailedProb = simParams.immediateCancelFailedProb,
		takingFailedProb = simParams.takingFailedProb,
		improvingFailedProb = simParams.improvingFailedProb,
		joinFailedProb = simParams.joinFailedProb,
		fill1MktVolumeFilter = simParams.fill1MktVolumeFilter,
		fill1VolumeConstraintFilter = simParams.fill1VolumeConstraintFilter,
		useOrderLog = simParams.useOrderLog,
		orderLogStructArray = orderLog,
		)

	## initialize 'TradeResults' named tuple with defauls
	results = TradeResults(
		trades = [Trade(refId = 0,
						of_flag = of_flag_.INIT,
						fillType = 1,
						tradePrice = 0.0,
						volume = 0,
						initVolume = 0,
						direction = direction_.SELL,
						orderSentTime = time[0],
						executeTime = time[0],
						submitType = -1,
						drivenType = -1,
						isDetecting = False,
						qBeforeTrade = -1.0,
						AS_OF_DATE = 0.0)],
						cancelNum = [0],
						profits=[(time[0],0.0)],
						)

	#### some pre-cleaning
	helper.activeBuyOrdersLocList.pop()
	helper.activeSellOrdersLocList.pop()
	helper.toCancelIDList.pop()
	helper.toCancelTimeList.pop() 
	helper.orderActionList.pop()
	
	results.profits.pop()
	results.trades.pop()

	#### initialize 
	queuePosHelper = helper.queuePosHelper[0]
	activeBuyOrdersLocList = helper.activeBuyOrdersLocList
	activeSellOrdersLocList = helper.activeSellOrdersLocList
	OrdersArray = helper.ordersArray
	myTopLevelQuantity = helper.myTopLevelQuantity
	previousCloseProfits = preSessionProfits



	#### ~~~ main loop ~~~ ####

	## iterate through each tick
	for i in range(dataLen):

		currentMkt = data[i]
		currentTime = time[i]
		currentInsRoot = insRoot[i]
		currentSignals =  signals[i]

		#### ~~~ handle data of traded instrument ~~~ ####
		if (currentInsRoot == tradeInsRoot):
			## simulate trade
			trades = simulateTrade(
						OrdersHelper = helper, 
						currentMkt = currentMkt,
						currentTime = currentTime,
                        context = context,
						simParams = simParams,
						)
			## determine open and close trades with fees 
			for trade in trades:
				## separate trades to open and close (of_flag)
				subTrades = update(Position = pos, trade = trade, OrdersHelper=helper)
				for subtrade in subTrades:
					## update position w/ trade costs
					updatePosition(trade = subtrade, Position = pos)
				## save trades to results
				results.trades.extend(subTrades)

			## trade feedback initiated orders
			for trade in trades:
				if trade.isDetecting:
					handleTrade(
						subTrade = trade,
						currentTime = currentTime,
						Position = pos,
						OrdersHelper = helper,
						context = context,
						params =params,
						)

			## record profits on condition		
			if abs(pos[0].closeProfits - previousCloseProfits) > 1e-7:
				results.profits.append((currentTime, pos[0].closeProfits))

			previousCloseProfits = pos[0].closeProfits

			## modify market based on our orders
			modifyMarket(
				currentMkt = data[i],
				OrdersArray = OrdersArray,
				activeBuyOrdersLocList = activeBuyOrdersLocList,
				activeSellOrdersLocList = activeSellOrdersLocList,
				myTopLevelQuantity = myTopLevelQuantity,
				flag = simParams.modifyMarket
				)

			## update queuePosHelper
			bidLen = context.signals[0].bidLen
			askLen = context.signals[0].askLen
			bidStack = context.bidStack
			askStack = context.askStack

			if bidLen > 2:
				bidQueueAfter = bidStack[bidLen-2] - bidStack[bidLen-3]
			elif bidLen > 1:
				bidQueueAfter = bidStack[bidLen-2]
			else:
				bidQueueAfter = np.nan
			if askLen > 2:
				askQueueAfter = askStack[askLen-2] - askStack[askLen-3]
			elif askLen > 1:
				askQueueAfter = askStack[askLen-2]
			else:
				askQueueAfter = np.nan

			queuePosHelper.bz = currentMkt.bz
			queuePosHelper.bz1 = bidQueueAfter
			queuePosHelper.az = currentMkt.az
			queuePosHelper.az1 = askQueueAfter
			queuePosHelper.bid = currentMkt.bid
			queuePosHelper.bid1 = currentMkt.bid-MINMOVE
			queuePosHelper.ask = currentMkt.ask
			queuePosHelper.ask1 = currentMkt.ask+MINMOVE

			## main strategy
			handleMktData(
				currentMkt = currentMkt, 
				currentTime = currentTime, 
				currentInsRoot = currentInsRoot,
				Position = pos,
				OrdersHelper = helper,
				currentSignals = currentSignals,
				context = context,
				params = params,
				multiplier = multiplier,
				MINMOVE = MINMOVE,
				)	
		#### ~~~ handle data of order return (NDni2012) ~~~ ####
		#else:# currentMkt.data_source == 43:
			#handleOrderReturnData(
				#currentMkt = currentMkt,
				#currentTime = currentTime, 
				#currentInsRoot = currentInsRoot,
				#Position = pos,
				#OrdersHelper = helper,
				#params = params,
				#context = context,
				#MINMOVE = MINMOVE,)            
		#### ~~~ handle data of assist instrument (signal) ~~~ ####
		else:
			handleSignalMktData(
				currentMkt = currentMkt, 
				currentTime = currentTime, 
				currentInsRoot = currentInsRoot,
				Position = pos,
				OrdersHelper = helper,
				currentSignals = currentSignals,
				context = context,
				params = params,
				)
        

		## record start of session
		if helper.sessionStartFlag[0]:
			helper.sessionStartFlag[0] = False

	## save cancled orders
	results.cancelNum[0] = len(set(helper.toCancelIDList))

	## final function returns
	return results, helper.orderActionList




#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~ function modify market data based on our orders ~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
@jit(nopython=True)
def modifyMarket(currentMkt,OrdersArray,activeBuyOrdersLocList,activeSellOrdersLocList,myTopLevelQuantity,flag):
	
	#### ~~~ <BUY orders> ~~~ ####
	if len(activeBuyOrdersLocList):
		#### determine myTopLevelQuantity
		topPrice = OrdersArray[activeBuyOrdersLocList[0]].limitPrice
		sizeAtTopPrice =0
		for i in range(len(activeBuyOrdersLocList)):
			oneOrder = OrdersArray[activeBuyOrdersLocList[i]]
			currentPrice = oneOrder.limitPrice
			if abs(currentPrice-topPrice)<0.0001:
				sizeAtTopPrice+=max(oneOrder.initVolume-oneOrder.tradedVolume,0)
			else:
				break
		myTopLevelQuantity[0] = sizeAtTopPrice
		#### modify market bid price and size
		if flag:
			if (currentMkt.bid - topPrice)<-0.0001:
				currentMkt.bid = topPrice
				currentMkt.bz = sizeAtTopPrice*1.0

			elif abs(currentMkt.bid - topPrice)<0.0001:
				currentMkt.bz += sizeAtTopPrice*1.0

	#### ~~~ <SELL orders> ~~~ ####
	if len(activeSellOrdersLocList):
		#### determine myTopLevelQuantity
		topPrice = OrdersArray[activeSellOrdersLocList[0]].limitPrice
		sizeAtTopPrice =0
		for i in range(len(activeSellOrdersLocList)):
			oneOrder = OrdersArray[activeSellOrdersLocList[i]]
			currentPrice = oneOrder.limitPrice
			if abs(currentPrice-topPrice)<0.0001:
				sizeAtTopPrice+=max(oneOrder.initVolume-oneOrder.tradedVolume,0)
			else:
				break
		myTopLevelQuantity[1] = sizeAtTopPrice
		#### modify market ask price and size
		if flag:
			if (currentMkt.ask - topPrice) >0.0001:
				currentMkt.ask = topPrice
				currentMkt.az = sizeAtTopPrice*1.0
			elif abs(currentMkt.ask - topPrice)<0.0001:
				currentMkt.az += sizeAtTopPrice*1.0



#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~ dummy placeholders for trading strategy functions ~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####

@jit(nopython=True)
def handleMktData(
	currentMkt, 
	currentTime, 
	currentInsRoot,
	Position,
	OrdersHelper,
	currentSignals,
	context,
	params,
	multiplier,	
	):
	pass

@jit(nopython=True)
def handleOrderReturnData(
	currentMkt,
	currentTime, 
	currentInsRoot,
	Position,
	OrdersHelper,
	params,
	context,
	MINMOVE,
	): 
	pass

@jit(nopython=True)
def handleSignalMktData(
	currentMkt, 
	currentTime, 
	currentInsRoot,
	Position,
	OrdersHelper,
	currentSignals,
	context,
	params,
	):
	pass

@jit(nopython=True)
def handleTrade(  
	subTrade,
	currentTime,
	Position,
	OrdersHelper,
	context,
	params,
	):
	pass


@jit(nopython=True)
def simulateTrade(  
	OrdersHelper,
	currentMkt,
	currentTime,
	):
	pass