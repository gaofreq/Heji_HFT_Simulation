from collections import namedtuple
import numpy as np
from numba.typed import List

from config import minMoveDict_
from strategyTemplate.strategyParams import SignalsDtype


## Context
Context = namedtuple('Context',
	['signals',
	 'bidStack',
	 'askStack',
	 'bidStackNoSpoofing',
	 'askStackNoSpoofing',
	 'circularBuff20',
	 'lastPriceHolder',
	 'bzHolder',
	 'azHolder',
	 'bidHolder',
	 'askHolder',
	 'volumeHolder',
	 'signalHolder',
	 'theoHolder',
	 'bidBook',
	 'askBook',
	])

#### ~~~ Params ~~~ ####
simParams = namedtuple('simParams', [
	## simulation-related
	'fastTriggerWt',
	'immediateCancelFailedProb',
	'takingFailedProb',
	'improvingFailedProb',
	'joinFailedProb',
	'modifyMarket',
	'fill1MktVolumeFilter',
	'fill1VolumeConstraintFilter',  
	'closeRate',
	'openRate',
	'QUEUEFRAC',
	'TRADEDVOLUMERATIO',
	'FILLEDVOLUMERATIO',
	'PERROUNDFEE',
	'saveOrders',
	'useOrderLog',
	'orderLagTime',
	'orderLeadTime',
	'dropoutRatio',
	])





simParamsInstance = simParams(
	## simulation-related
	fastTriggerWt = 0.75,
	immediateCancelFailedProb = 0.0,
	takingFailedProb = 0.0,
	improvingFailedProb = 0.0,
	joinFailedProb = 0.0,
	modifyMarket = False,
	fill1MktVolumeFilter = True,
	fill1VolumeConstraintFilter = True,
	closeRate = 0,
	openRate = 0,
	QUEUEFRAC = 0.0,
	TRADEDVOLUMERATIO = 1.0,
	FILLEDVOLUMERATIO=0.2,   
	PERROUNDFEE = 1.5/10.0*0.62,
	saveOrders = True,
	useOrderLog = False,
	orderLagTime = np.timedelta64(100000000, 'ns'),          # 100 micro seconds,0.1s
	orderLeadTime = np.timedelta64(100000000, 'ns'),          # 100 micro seconds, 0.1s
	dropoutRatio = -1.0,
	)



def ContextInstance(data, tradeInsRoot, verbose=False):

	data_temp = data.loc[data['instrument_root']==tradeInsRoot,["bid","ask"]]
	bidStackCapacity = int((data_temp["bid"].max() - data_temp["bid"].min()) / minMoveDict_[tradeInsRoot]) + 30
	askStackCapacity = int((data_temp["ask"].max() - data_temp["ask"].min()) / minMoveDict_[tradeInsRoot]) + 30

	if verbose:
		print("bidStackSize: " + str(bidStackCapacity))
		print("askStackSize: " + str(askStackCapacity))

	## create numba typed list (deprecation of reflected list)
	bidStack_list = [1.0]*bidStackCapacity
	askStack_list = [1.0]*askStackCapacity
	bidStackNoSpoofing_list = [1.0]*bidStackCapacity
	askStackNoSpoofing_list = [1.0]*askStackCapacity
	typed_bidStack_list = List()
	typed_askStack_list = List()
	typed_bidStackNoSpoofing_list = List()
	typed_askStackNoSpoofing_list = List()
	[typed_bidStack_list.append(x) for x in bidStack_list]
	[typed_askStack_list.append(x) for x in askStack_list]
	[typed_bidStackNoSpoofing_list.append(x) for x in bidStackNoSpoofing_list]
	[typed_askStackNoSpoofing_list.append(x) for x in askStackNoSpoofing_list]
    
	bidBook_list = np.zeros(100)
	askBook_list = np.zeros(100)

	ans = Context(
		np.zeros(1,dtype=SignalsDtype),
		typed_bidStack_list,
		typed_askStack_list,
		typed_bidStackNoSpoofing_list,
		typed_askStackNoSpoofing_list,
		np.zeros(20,dtype='float64'),
		np.zeros(40,dtype='float64'),
		np.zeros(40,dtype='float64'),
		np.zeros(40,dtype='float64'),
		np.zeros(40,dtype='float64'),
		np.zeros(40,dtype='float64'),
		np.zeros(40,dtype='float64'),
		np.zeros(40,dtype='float64'),
		np.zeros(40,dtype='float64'),
        bidBook_list,
        askBook_list,
		)

	return(ans)