
from __future__ import print_function
from collections import namedtuple
import numpy as np
from tqdm import tqdm
import sys

from library.simulator.simulationForSession import simulationForSession
from library.simulator.types import orderLogDtype
import library.simulator.simulationForSession as simuSession 
from config import volMultipleDict_, minMoveDict_

		


#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~ function to run simulation (session-level data) ~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####

def backtestPartial(dataInput,
					params,
					simParams,  
					signals,
					context, 
					insRootDict,
					tradeInsRoot,
					staticProfits,
					strategyModule,
					orderManagerModule,
					dataLenCut,
					orderLogInput = None,
					verbose = True,
					):  

	## placeholders for empty trading strategy functions
	handleMktDataEmpty = simuSession.handleMktData
	#handleOrderReturnDataEmpty = simuSession.handleOrderReturnData
	handleSignalMktDataEmpty = simuSession.handleSignalMktData
	handleTradeEmpty = simuSession.handleTrade
	simulateTradeEmpty = simuSession.simulateTrade

	try:	
		## import trading strategy functions from strategy module
		exec("from strategyTemplate.{0} import handleMktData, handleSignalMktData, handleTrade".format(strategyModule))   # rm: handleTrade
		exec("simuSession.handleMktData = handleMktData")
		#exec("simuSession.handleOrderReturnData = handleOrderReturnData")
		exec("simuSession.handleSignalMktData = handleSignalMktData")
		exec("simuSession.handleTrade = handleTrade")		# rm: handleTrade
		exec("from library.simulator.{0} import simulateTrade, submitOrder, cancelOrder".format(orderManagerModule))
		exec("simuSession.simulateTrade = simulateTrade")
		exec("import strategyTemplate.{0} as stratModule".format(strategyModule))
		exec("stratModule.submitOrder = submitOrder")
		exec("stratModule.cancelOrder = cancelOrder")
		
		#### prepare data for simulation
		data = dataInput.copy()
		data['vwap'] = data['turnover'] /data['volume'] / (data['instrument_root'].map(volMultipleDict_))
		data.loc[data.volume < 1, 'vwap'] = np.nan
		
		time = data['machine_time'].values
		insRoot = data['instrument_root'].map(insRootDict).values
		datafloatBlock = data[data.dtypes.index[data.dtypes==np.float64]]
		dtypeData = [(str(x),np.float64) for x in datafloatBlock.columns]
		dataReady = np.recarray(shape = ( len(data),), buf = datafloatBlock.values.copy(), dtype=dtypeData,)

		## zero signals if not given 
		if signals is None:
			signals = np.zeros(len(data),dtype='float64')

		sessionArray = data.groupby('session', sort=False).size().cumsum().values  ## an array containing length of each session
		tmpStaticProfits = staticProfits
		## placeholders to return
		results = []
		orders = []
		countFlag = False 


		#### iterate through sessions
		for j in tqdm(range(0, len(sessionArray)), ncols=69, disable=(not verbose)):
			
			## dropout
			if np.random.uniform(0.0,1.0) < simParams.dropoutRatio:
				continue

			if j:
				k = sessionArray[j-1]
			else:
				k = 0
			while insRoot[k]!=insRootDict[tradeInsRoot]:
				k = k + 1                    
			start = k
			

			if j < len(sessionArray)-1:
				m = sessionArray[j]
			else:
				m = sessionArray[j]                
			while insRoot[m-1]!=insRootDict[tradeInsRoot]:
				m = m - 1                    
			end = m
			
			if (end-start)<=dataLenCut:
				continue
			if countFlag:
				if results[-1].profits:
					tmpStaticProfits = (results[-1].profits)[-1][1]
			
			countFlag = True 
			## prepare orderLogReady 
			time_min = time[start:end].min()
			time_max = time[start:end].max()
			if simParams.useOrderLog:
				if orderLogInput is None: sys.exit('Order Log is needed if useOrderLog!')
				orderLogReady = orderLogInput.loc[(orderLogInput["MACHINE_TIMESTAMP"]>=time_min)&(orderLogInput["MACHINE_TIMESTAMP"]<=time_max),:].copy()
				if len(orderLogReady) == 0:
					orderLogReady = np.empty(1, dtype=orderLogDtype)
				else:
					orderLogReady["INDEX"] = range(len(orderLogReady))  ## add INDEX to identify USED or not
					orderLogReady = orderLogReady.to_records(index=False)
			else:
				orderLogReady = np.empty(1, dtype=orderLogDtype) 

			sessionResult, orderActions = simulationForSession (data = dataReady[start:end],
																time = 	time[start:end], 
																insRoot = insRoot[start:end],
																signals = signals[start:end],
																params = params,
																simParams = simParams,
																context = context,
																preSessionProfits = tmpStaticProfits,
																tradeInsRoot = insRootDict[tradeInsRoot],
																multiplier = volMultipleDict_[tradeInsRoot],
																MINMOVE = minMoveDict_[tradeInsRoot],
																orderLog = orderLogReady,
																sessionNum = int(data["session"].tolist()[start][-1]),
																)
			## save results
			results.append(sessionResult)
			if simParams.saveOrders:
				orders.append(orderActions)

	except:
		raise
	
	finally:
		## resume to orignal empty functions
		simuSession.handleMktData = handleMktDataEmpty
		#simuSession.handleOrderReturnData = handleOrderReturnDataEmpty
		simuSession.handleSignalMktData = handleSignalMktDataEmpty
		simuSession.handleTrade = handleTradeEmpty
		simuSession.simulateTrade = simulateTradeEmpty

	## return params w/ or w/o orders 
	if simParams.saveOrders:
		return results, orders
	else:
		return results
 

