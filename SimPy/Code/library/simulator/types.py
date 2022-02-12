import numpy as np
from collections import namedtuple



## Position
PositionDtype = np.dtype([
	('shortPosition', np.int64), 
	('longPosition', np.int64),
	('longAveragePrice', np.double),
	('shortAveragePrice', np.double),
	('closeProfits', np.double),
	('PERROUNDFEE', np.double),
	('openRate', np.double),
	('closeRate', np.double),
	])


## Order 
OrderDtype = np.dtype([
	('direction', np.int8), 
	('refId', np.int64),
	('initVolume', np.int64),
	('tradedVolume', np.int64),
	('queueFrontFixed', np.double),
	('queueFrontJudge', np.double),
	('posJudge', np.double),
	('sentTime','<M8[ns]'),
	('firstFlag', np.bool),
	('deepQuoteFlag', np.bool),
	('orderStatus', np.int8),
	('limitPrice', np.double),
	('submitType', np.int64),
	('drivenType', np.int64),
	('isTriggered', np.bool),
	('isDetecting', np.bool),
	('isDetectingTriggered', np.bool),
	('isOrderLogStatus', np.bool),
	('orderLogRefId',np.int64),
	('isOrderLogCancelStatus', np.bool),
	('isOpen', np.bool),
	])


## QueuePosHelper 
QueuePosHelperDtype = np.dtype([
	('bz', np.double),
	('az', np.double),
	('bid', np.double),
	('ask', np.double),
	('bz1', np.double),
	('az1', np.double),
	('bid1', np.double),
	('ask1', np.double)                   
	])

## orderLog
orderLogDtype = np.dtype([
	('REF_ID', np.int64),
	('LOG_NAME', np.int64),
	('MACHINE_TIMESTAMP', '<M8[ns]'),
	('VOLUME', np.double),
	('CURRENT_VOLUME', np.double),
	('DIRECTION', np.int64),
	('LIMIT_PRICE', np.double),
	('STATUS', np.int64),
	('TRADED_VOLUME', np.double),
	('USED', np.int64),
	('INSERT_MACHINE_TIMESTAMP', '<M8[ns]'),
	('INDEX', np.int64),
	])



Trade = namedtuple('Trade',	[
	'refId',
	'direction',
	'of_flag',
	'volume',
	'initVolume',
	'tradePrice',
	'fillType',
	'orderSentTime',
	'executeTime',
	'submitType',
	'drivenType',
	'isDetecting',
	'qBeforeTrade',
	'AS_OF_DATE'
	])


#refId and sessionStartFlag  are one element lists 
OrdersHelper = namedtuple('OrdersHelper', [
	'activeBuyOrdersLocList',
	'activeSellOrdersLocList',
	'myTopLevelQuantity',
	'slotsLocList',    
	'refId',
	'queuePosHelper',
	'toCancelIDList',
	'toCancelTimeList',
	'ordersArray',
	'sessionStartFlag',
	'sessionLastTime',
	'eps',
	'MINMOVE',
	'QUEUEFRAC',
	'fastTriggerWt',
	'TRADEDVOLUMERATIO',
	'FILLEDVOLUMERATIO',
	'saveOrders',
	'orderActionList',
	'immediateCancelFailedProb',
	'takingFailedProb',
	'improvingFailedProb',
	'joinFailedProb',
	'fill1MktVolumeFilter',
	'fill1VolumeConstraintFilter',
	'useOrderLog',
	'orderLogStructArray',
	])


#cancelNum is a one element list
TradeResults = namedtuple('TradeResults', [
	'trades',
	'cancelNum',
	'profits',
	])


#cancelNum is a one element list
OrderAction = namedtuple('OrderAction', [
	'refId',
	'actionTime',
	'initVolume',
	'cancelling',
	'direction',
	'limitPrice',
	'detRefId',
	'isDetecting',
	'longPosition',
	'shortPosition',
	'outStandingBuy',
	'outStandingSell',
	'submitType',
	'drivenType',
	'isOrderLogMatched',
	'orderLogRefId',
	'AS_OF_DATE',
	])