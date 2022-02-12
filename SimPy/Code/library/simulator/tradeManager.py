from numba import jit
import numpy as np

from config import direction_, of_flag_
from library.simulator.types import Trade



#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~ function to separate trades by of_flag ~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####

@jit(nopython=True, cache=True)
def update(Position, trade, OrdersHelper):

	#### initializations
	trades = []
	position = Position[0]

	#### ~~~ calculate buyOrderTotalCount and sellOrderTotalCount ~~~ ####
	ordersArray = OrdersHelper.ordersArray

	buyOrderTotalCount = 0
	for loc in OrdersHelper.activeBuyOrdersLocList:
		order = ordersArray[loc]
		if order.isOpen==False:
			buyOrderTotalCount += np.maximum(order.initVolume-order.tradedVolume,0)

	sellOrderTotalCount = 0
	for loc in OrdersHelper.activeSellOrdersLocList:
		order = ordersArray[loc]
		if order.isOpen==False:
			sellOrderTotalCount += np.maximum(order.initVolume-order.tradedVolume,0)

	#### ~~~ <BUY trades> ~~~ ####
	if (trade.direction == direction_.BUY):
		shortNetPosition = position.shortPosition - buyOrderTotalCount
		#### <A> non-zero short position to be closed
		if (shortNetPosition > 0):
			#### <A.1> trade is enough to close current and open new position
			if (trade.volume - shortNetPosition > 0):
				## sub-trade to close current position
				trade1 = Trade (
					refId = trade.refId,
					of_flag = of_flag_.CLOSE,
					fillType = trade.fillType,
					tradePrice = trade.tradePrice,
					volume = shortNetPosition,
					direction = direction_.BUY,
					orderSentTime = trade.orderSentTime,
					executeTime = trade.executeTime,
					submitType = trade.submitType,
					drivenType = trade.drivenType,  
					initVolume = trade.initVolume,
					isDetecting = trade.isDetecting,
					qBeforeTrade = trade.qBeforeTrade,
					AS_OF_DATE = trade.AS_OF_DATE,
					)
				trades.append(trade1)

				## sub-trade to open new position
				trade2 = Trade (
					refId = trade.refId,
					of_flag = of_flag_.OPEN,
					fillType = trade.fillType,
					tradePrice = trade.tradePrice,
					volume = trade.volume-shortNetPosition,
					direction = direction_.BUY,
					orderSentTime = trade.orderSentTime,
					executeTime = trade.executeTime,
					submitType = trade.submitType,
					drivenType = trade.drivenType,
					initVolume = trade.initVolume,
					isDetecting = trade.isDetecting,
					qBeforeTrade = trade.qBeforeTrade,
					AS_OF_DATE = trade.AS_OF_DATE,
					)
				trades.append(trade2)

			#### <A.2> trade is not enough to open new position
			elif (trade.volume - shortNetPosition <= 0):
				## sub-trade to close current position
				trade1 = Trade (
					refId = trade.refId,
					of_flag = of_flag_.CLOSE,
					fillType = trade.fillType,
					tradePrice = trade.tradePrice,
					volume = trade.volume,
					direction = direction_.BUY,
					orderSentTime = trade.orderSentTime,
					executeTime = trade.executeTime,
					submitType = trade.submitType,
					drivenType = trade.drivenType,
					initVolume = trade.initVolume,
					isDetecting = trade.isDetecting,
					qBeforeTrade = trade.qBeforeTrade,
					AS_OF_DATE = trade.AS_OF_DATE,
					)
				trades.append(trade1)

		#### <B> no short position to be closed
		elif shortNetPosition <= 0:
			## sub-trade to open new position
			trade1 = Trade (
				refId = trade.refId,
				of_flag = of_flag_.OPEN,
				fillType = trade.fillType,
				tradePrice = trade.tradePrice,
				volume = trade.volume,
				direction = direction_.BUY,
				orderSentTime = trade.orderSentTime,
				executeTime = trade.executeTime,
				submitType = trade.submitType,
				drivenType = trade.drivenType,
				initVolume = trade.initVolume,
				isDetecting = trade.isDetecting,
				qBeforeTrade = trade.qBeforeTrade,
				AS_OF_DATE = trade.AS_OF_DATE,
				)
			trades.append(trade1)


	#### ~~~ <SELL trades> ~~~ ####
	elif (trade.direction == direction_.SELL):
		longNetPosition = position.longPosition - sellOrderTotalCount
		#### <A> non-zero long position to be closed
		if (longNetPosition > 0):
			#### <A.1> trade is enough to close current and open new position
			if (trade.volume - longNetPosition > 0):
				## sub-trade to close current position
				trade1 = Trade (
					refId = trade.refId,
					of_flag = of_flag_.CLOSE,
					fillType = trade.fillType,
					tradePrice = trade.tradePrice,
					volume = longNetPosition,
					direction = direction_.SELL,
					orderSentTime = trade.orderSentTime,
					executeTime = trade.executeTime,
					submitType = trade.submitType,
					drivenType = trade.drivenType,
					initVolume = trade.initVolume,
					isDetecting = trade.isDetecting,
					qBeforeTrade = trade.qBeforeTrade,
					AS_OF_DATE = trade.AS_OF_DATE,
					)
				trades.append(trade1)

				## sub-trade to open new position
				trade2 = Trade (
					refId = trade.refId,
					of_flag = of_flag_.OPEN,
					fillType = trade.fillType,
					tradePrice = trade.tradePrice,
					volume = trade.volume-longNetPosition,
					direction = direction_.SELL,
					orderSentTime = trade.orderSentTime,
					executeTime = trade.executeTime,
					submitType = trade.submitType,
					drivenType = trade.drivenType,
					initVolume = trade.initVolume,
					isDetecting = trade.isDetecting,
					qBeforeTrade = trade.qBeforeTrade,
					AS_OF_DATE = trade.AS_OF_DATE,
					)
				trades.append(trade2)

			#### <A.2> trade is not enough to open new position
			elif (trade.volume - longNetPosition <= 0):
				## sub-trade to close current position
				trade1 = Trade (
					refId = trade.refId,
					of_flag = of_flag_.CLOSE,
					fillType = trade.fillType,
					tradePrice = trade.tradePrice,
					volume = trade.volume,
					direction = direction_.SELL,
					orderSentTime = trade.orderSentTime,
					executeTime = trade.executeTime,
					submitType = trade.submitType,
					drivenType = trade.drivenType,
					initVolume = trade.initVolume,
					isDetecting = trade.isDetecting,
					qBeforeTrade = trade.qBeforeTrade,
					AS_OF_DATE = trade.AS_OF_DATE,
					)
				trades.append(trade1)

		#### <B> no short position to be closed
		elif longNetPosition <= 0:
			trade1 = Trade (
				refId = trade.refId,
				of_flag = of_flag_.OPEN,
				fillType = trade.fillType,
				tradePrice = trade.tradePrice,
				volume = trade.volume,
				direction = direction_.SELL,
				orderSentTime = trade.orderSentTime,
				executeTime = trade.executeTime,
				submitType = trade.submitType,
				drivenType = trade.drivenType,
				initVolume = trade.initVolume,
				isDetecting = trade.isDetecting,
				qBeforeTrade = trade.qBeforeTrade,
				AS_OF_DATE = trade.AS_OF_DATE,
				)
			trades.append(trade1)

	#### final funcion return
	return trades




#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~ function to update position w/ trade costs by of_flag ~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #### 

@jit(nopython=True, cache=True)
def updatePosition(trade, Position):

	#### initializations
	position = Position[0]
	PERROUNDFEE =position.PERROUNDFEE
	openRate = position.openRate
	closeRate = position.closeRate

	#### ~~~ <BUY trades> ~~~ ####
	if (trade.direction == direction_.BUY):
		#### trade to close position 
		if (trade.of_flag == of_flag_.OPEN):
			## update
			position.longAveragePrice = (position.longPosition*position.longAveragePrice + trade.volume*trade.tradePrice*(1.0+openRate)) / (position.longPosition + trade.volume)
			position.longPosition += trade.volume
		#### trade to close position
		elif (trade.of_flag == of_flag_.CLOSE):
			## update
			position.shortPosition -= trade.volume
			position.closeProfits += trade.volume * (position.shortAveragePrice - trade.tradePrice*(1.0+closeRate) - PERROUNDFEE)

	### ~~~ <SELL trades> ~~~ ####
	elif (trade.direction == direction_.SELL):
		#### trade to close position 
		if (trade.of_flag == of_flag_.OPEN):
			## update
			position.shortAveragePrice = (position.shortPosition * position.shortAveragePrice + trade.volume * trade.tradePrice*(1.0-openRate)) / (position.shortPosition + trade.volume)
			position.shortPosition += trade.volume
		#### trade to close position
		elif (trade.of_flag == of_flag_.CLOSE):
			## update
			position.longPosition -= trade.volume
			position.closeProfits += trade.volume * (trade.tradePrice*(1.0-closeRate) - position.longAveragePrice-PERROUNDFEE)
