#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~~ class: StatsCompare() ~~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####

import gc 
import pandas as pd
from itertools import product 

from library.simulator.PnlAdjust import PnlAdjust
from library.simParams import ContextInstance


class StatsCompare(object):

	def __init__(self, dataInputGen, paramsList, paramNames, simParamsList, simParamNames):

		## all parameters must have values in input (even if not used)
		assert len(paramNames) == len (paramsList) and len(simParamNames) == len (simParamsList)

		#### define attrbutes with default values
		self.dataInputGen = dataInputGen
		self.paramNames = paramNames
		self.simParamNames = simParamNames
		self.paramsMergeList = paramsList + simParamsList
		self.paramsMergeCombination = list(product(*self.paramsMergeList))	
		self.paramComboDictList = [dict(zip(names,values)) for names,values in zip([paramNames+simParamNames]*len(self.paramsMergeCombination), self.paramsMergeCombination)]
		
		self.tradeNumList = []
		self.joiningTradeNumList = []
		self.improvingTradeNumList = []
		self.takingTradeNumList = []
		self.cancelNumList = []
		self.terminalList = [] 
		self.feeList = []
		self.finalAdjustedFrameListByParams  = []
		self.tradeParticipateListByParams = []
		self.tradesFrameListOfListByParams = []
		self.ordersFrameListOfListByParams = []
		self.dailyPnlByParams = []
		self.sessionFrame = []
		self.paramsComboDictListofList = []
		self.paramsFullDictList = []
		self.statsResults = None 


	#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
	#### ~~~ function to run simulation and save results ~~~ ####
	#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
	def crossCompare(self, 
					params,
					simParams,
					insRootDict, 
					tradeInsRoot,
					strategyModule,
					orderManagerModule, 
					dataLenCut = 10,
					signalName = None ,
					openFee = 0.0,
					closeFee = 0.0,
					openRate = 0.0,
					closeRate = 0.0,
					verbose = True,
					):

		if verbose:
			print("Trading instrument: '" + tradeInsRoot + "'")
			print("Using strategy module: '" + strategyModule + "'")
			print("Using orderManager module: '" + orderManagerModule + "'")

		cumLen = 0
		session = []
		sessionLen = []

		#### iterate through backtesting data sections
		for key in [x for x in self.dataInputGen.keys() if "MKT_DATA" in x]:

			data = self.dataInputGen[key]
			orderLog = self.dataInputGen[key.replace("MKT_DATA","ORDER_LOG")] if key.replace("MKT_DATA","ORDER_LOG") in self.dataInputGen.keys() else None
			context = ContextInstance(data, tradeInsRoot)

			if signalName is not None:
				toUseSignals = data[signalName].values[cumLen:(cumLen+len(data))]
			else:
				toUseSignals = None 
			cumLen+= len(data)

			session.append(data.session.unique())
			sessionLen.append(data.groupby('session').size())
			pnl= PnlAdjust (dataInput = data,
							params = params,
							simParams = simParams,
							context = context,       
							tradeInsRoot = tradeInsRoot,
							insRootDict = insRootDict,
							strategyModule = strategyModule,
							orderManagerModule = orderManagerModule,
							signals = toUseSignals,
							dataLenCut = dataLenCut,
							orderLogInput = orderLog,
							)
			terminal = []
			tradeNum = []
			joiningTradeNum = []
			improvingTradeNum = []
			takingTradeNum = []
			cancelNum = [] 
			paramsDictList = []
			fee = []
			tradesListByParams = []
			ordersListByParams = []

			
			## loop through merge combination of params
			for paramComboDict in self.paramComboDictList:

				if verbose:
					print("Parameter set " + str(self.paramComboDictList.index(paramComboDict)+1) + " out of " + str(len(self.paramComboDictList)) + ":")

				## overwrite paramsInstance with input values
				paramDict = {k: paramComboDict[k] for k in self.paramNames}
				simParamDict = {k: paramComboDict[k] for k in self.simParamNames}
				pnl.reset(params._replace(**paramDict), simParams._replace(**simParamDict))
				if verbose:
					print("Using order log: " + str(simParams._replace(**simParamDict).useOrderLog))

				## compute pnl
				pnl.computePnl(openFee,closeFee,openRate,closeRate,verbose)

				## apend list from attributes of PnlAdjust
				terminal.append(pnl.terminalFrame)
				tradeNum.append(pnl.tradeNumFrame)
				joiningTradeNum.append(pnl.joiningTradeNumFrame)
				improvingTradeNum.append(pnl.improvingTradeNumFrame)
				takingTradeNum.append(pnl.takingTradeNumFrame)
				cancelNum.append(pnl.cancelNumFrame)
				fee.append(pnl.feeFrame)
				tradesListByParams.append(pnl.tradesFrameList)
				ordersListByParams.append(pnl.ordersFrameList)
				paramsDictList.append(paramComboDict)  ## not from PnlAdjust
				
				## update and record paramsFullDictList
				self.paramsFullDictList.append({**params._replace(**paramDict)._asdict(), **simParams._replace(**simParamDict)._asdict()})

				## clear memory
				gc.collect()

			#### save to attributes
			self.terminalList.append(terminal)
			self.tradeNumList.append(tradeNum)
			self.joiningTradeNumList.append(joiningTradeNum)
			self.improvingTradeNumList.append(improvingTradeNum)
			self.takingTradeNumList.append(takingTradeNum)
			self.cancelNumList.append(cancelNum)
			self.feeList.append(fee)
			self.tradesFrameListOfListByParams.append(tradesListByParams)
			self.ordersFrameListOfListByParams.append(ordersListByParams)
			self.paramsComboDictListofList.append(paramsDictList)

		## update sessionFrame for simulation results
		sessionLenFrame = pd.concat(sessionLen,ignore_index=True)
		self.sessionFrame = pd.concat([pd.DataFrame(x) for x in session],ignore_index=True)
		self.sessionFrame = self.sessionFrame.loc[sessionLenFrame>dataLenCut,:]
		self.sessionFrame.index=range(len(self.sessionFrame))
		self.sessionFrame.columns=['session']



	#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
	#### ~~~ function to calculate rates for different order submitTypes ~~~ ####
	#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
	def getTradeStats(self):

		if not self.tradeParticipateListByParams:
			joiningTradeNumList = zip(*self.joiningTradeNumList)
			improvingTradeNumList = zip(*self.improvingTradeNumList)
			takingTradeNumList = zip(*self.takingTradeNumList)
			tradeNumList = zip(*self.tradeNumList)

			for joiningTradeNum, improvingTradeNum,takingTradeNum,tradeNum in zip(joiningTradeNumList,improvingTradeNumList,takingTradeNumList,tradeNumList):
				joinSum = pd.concat(joiningTradeNum).sum()
				improvingSum = pd.concat(improvingTradeNum).sum()
				takingSum = pd.concat(takingTradeNum).sum()
				totalSum = pd.concat(tradeNum).sum()*1.0

				joinRate = joinSum/totalSum
				improvingRate = improvingSum/totalSum
				takingRate = takingSum/totalSum

				rateFrame = pd.concat([joinRate,improvingRate,takingRate],axis=1)
				rateFrame.columns = ['join','improving','taking']

				self.tradeParticipateListByParams.append(rateFrame)



	#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
	#### ~~~~ function to summarize trade stats ~~~~ ####
	#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
	def getStats(self, cancelFailWeight,takingFailWeight,improvingFailWeight,joinFailWeight,allFailWeight):

		 
		tradeNumList = zip(*self.tradeNumList)
		cancelNumList = zip(*self.cancelNumList)
		terminalList = zip(*self.terminalList)
		self.finalAdjustedFrameListByParams = []
		self.dailyPnlByParams = []
		

		sessionFrame = self.sessionFrame

		statsList = []

		for tradeNum,cancelNum,terminal in zip(tradeNumList,cancelNumList,terminalList):

			shifted = []
			pre = 0.0

			cumTerminal = 0.0
			
		  
			for x in terminal:

				cumTerminal= cumTerminal+x.iloc[-1,:]
				adjusted = x.allSuccess-(cancelFailWeight*x.cancelDiff+
								   takingFailWeight*x.takingDiff+
								   improvingFailWeight*x.improvingDiff+
								   joinFailWeight*x.joinDiff+
                                   allFailWeight*x.allDiff)
				shifted.append(pre+adjusted)
				tmp = adjusted.iloc[-1]
				pre += tmp 


			final =pd.concat(shifted,ignore_index=True)

			self.finalAdjustedFrameListByParams.append(final)

	   

			final = final.fillna(method='ffill')

			sessionly=final.diff()

			sessionly.iloc[0]=final.iloc[0]

			sessionly.index=range(len(sessionly))

			realDaily=sessionly.groupby(sessionFrame.session.map(lambda x: x[2:10]),sort=False).sum()

			self.dailyPnlByParams.append(realDaily)

			sharpe = realDaily.mean()/realDaily.std()*15

			sessionlyWinningProb = ((sessionly>0.0).sum())*1.0/len(sessionly)

			maxDrawDown=(final-final.cummax()).min()  ## Tongchuan edited

			maxDailyLoss = realDaily.min()

			#### prepare for statsList
			statsSummary = pd.DataFrame([sharpe,final.iloc[-1],maxDrawDown,sessionlyWinningProb,maxDailyLoss],
									   index = ['sharpe','terminal','sessionlyDrawDown','sessionlyWinningProb','maxDailyLoss'],
									   columns=['statsValue'])
			## tradeNumFrame
			tradeNumFrame = pd.concat(tradeNum).sum().T
			tradeNumFrame = pd.DataFrame(tradeNumFrame)
			tradeNumFrame.index= tradeNumFrame.index.map(lambda x:x+'TradeNum')
			tradeNumFrame.columns = statsSummary.columns
			## cancelNumFrame
			cancelNumFrame = pd.concat(cancelNum).sum().T
			cancelNumFrame = pd.DataFrame(cancelNumFrame)
			cancelNumFrame.index = cancelNumFrame.index.map(lambda x:x+'CancelNum')
			cancelNumFrame.columns = statsSummary.columns
			## terminalFrame
			terminalFrame = pd.DataFrame(cumTerminal)
			terminalFrame.columns = statsSummary.columns
			terminalFrame.index= terminalFrame.index.map(lambda x:x+'Terminal')
			## 	statsList
			statsList.append(pd.concat([statsSummary,tradeNumFrame,cancelNumFrame,terminalFrame]))

		#### save to attribute 'statsResults'
		self.statsResults = pd.concat(statsList,axis=1).T
		self.statsResults.index=range(len(self.statsResults))
		self.statsResults = pd.concat([self.statsResults, pd.DataFrame(self.paramsComboDictListofList[0])],axis=1)
		
