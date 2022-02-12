#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~~ class: PnlAdjust() ~~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####

import pandas as pd

from library.simulator.entryPoint import backtestPartial
from config import submitType_
from library.simulator.util import computeFee


class PnlAdjust(object):
	def __init__(self,			
				dataInput, 
				params,
				simParams,
				context,
				insRootDict, 
				tradeInsRoot,
				strategyModule,
				orderManagerModule,
				dataLenCut,
				signals = None,
				orderLogInput = None,
				):

		#### save arguments to attributes
		self.dataInput = dataInput
		self.orderLogInput = orderLogInput
		self.params = params
		self.simParams= simParams
		self.context = context
		self.insRootDict =  insRootDict
		self.tradeInsRoot = tradeInsRoot
		self.strategyModule =  strategyModule
		self.orderManagerModule = orderManagerModule
		self.dataLenCut =  dataLenCut    
		self.signals = signals

		#### define attributes with default values
		self.allGoodParams = simParams._replace(**{'immediateCancelFailedProb':0.0,
													'takingFailedProb':0.0,
													'improvingFailedProb':0.0,
													'joinFailedProb':0.0})
		self.terminalFrame = None 
		self.tradeNumFrame = None
		self.cancelNumFrame = None 
		self.joiningTradeNumFrame = None 
		self.improvingTradeNumFrame = None 
		self.takingTradeNumFrame = None
		self.feeFrame = None
		self.finalAdjustedFrame = None  
		self.tradesFrameList = []
		self.ordersFrameList = []
		self.saveOrders = self.simParams.saveOrders


				 

	#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####		
	#### ~~~ class function to reset parameter values to default ~~~ ####
	#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####		
	def reset(self, params, simParams):

		#### save arguments to attributes
		self.params = params
		self.simParams = simParams

		#### set default values to attributes
		self.allGoodParams = simParams._replace(**{'immediateCancelFailedProb':0.0,
												 'takingFailedProb':0.0,
												 'improvingFailedProb':0.0,
												 'joinFailedProb':0.0})
		self.terminalFrame = None 
		self.tradeNumFrame = None
		self.cancelNumFrame = None 
		self.joiningTradeNumFrame = None 
		self.improvingTradeNumFrame = None
		self.takingTradeNumFrame = None 
		self.feeFrame = None
		self.finalAdjustedFrame = None 
		self.tradesFrameList = []
		self.ordersFrameList = []
		self.saveOrders = self.simParams.saveOrders




	#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
	#### ~~~ class function to run simulation and return results ~~~ ####
	#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #### 
	def computePnl (self,
					openFee = 0.0,
					closeFee = 0.0,
					openRate = 0.0,
					closeRate = 0.0,
					verbose = True,
					):

		if self.terminalFrame is None or self.tradeNumFrame is None or self.cancelNumFrame is None or self.feeFrame is None:

			## scenarios to be iterated through		
			replaceDictList = [ {},
								{'immediateCancelFailedProb':1.0},
								{'takingFailedProb':1.0},
								{'improvingFailedProb':1.0},
								{'joinFailedProb':1.0},
								{'immediateCancelFailedProb':1.0,'takingFailedProb':1.0,'improvingFailedProb':1.0,'joinFailedProb':1.0},
							  ]

			## lists to be appended	  
			terminalProfitsList = []
			tradeNumList = []
			joiningTradeNumList = []
			improvingTradeNumList = []
			takingTradeNumList = []
			cancelNumList = []
			feeList = []
			
		 
			#### ~~~ iterate through scenarios ~~~ ####
			for replaceDict in replaceDictList:
				if verbose:                  
					print("Scenario " + str(replaceDictList.index(replaceDict)+1) + " out of " + str(len(replaceDictList)) + ":")
				
				## call of backtest
				tradeReAndOrders = backtestPartial (dataInput = self.dataInput,
													params = self.params,
													simParams = self.allGoodParams._replace(**replaceDict),
													context = self.context,
													signals = self.signals,
													tradeInsRoot = self.tradeInsRoot,
													staticProfits = 0.0,
													insRootDict = self.insRootDict,
													strategyModule = self.strategyModule,
													orderManagerModule = self.orderManagerModule,
													dataLenCut = self.dataLenCut,
													orderLogInput = self.orderLogInput,
													verbose = verbose,
													)

				#### save w/ or w/o orders
				if self.saveOrders:
					tradeRe = tradeReAndOrders[0]
				else:
					tradeRe = tradeReAndOrders
						
			 
				#### ~~~ prepare trade results to be appened to lists ~~~ ####

				## tradeNum, joiningTradeNum, improvingTradeNum, takingTradeNum 
				tradeNum = [sum(trade.volume for trade in session.trades) for session in tradeRe]
				joiningTradeNum = [sum(trade.volume if trade.submitType == submitType_.JOINING else 0 for trade in session.trades) for session in tradeRe]
				improvingTradeNum = [sum(trade.volume if trade.submitType == submitType_.IMPROVING else 0 for trade in session.trades) for session in tradeRe]
				takingTradeNum = [sum(trade.volume if trade.submitType == submitType_.TAKING else 0 for trade in session.trades) for session in tradeRe]
				## tradesFrameList (append done here)
				trades = []
				for session in tradeRe:
					trades+= session.trades
				self.tradesFrameList.append(trades)
				## ordersFrameList (append done here)
				if self.saveOrders:
					orders = []
					simuOrders = tradeReAndOrders[1]
					for session in simuOrders:
						orders+= session
					self.ordersFrameList.append(orders)  
				## cancelNumList
				cancelNum =[ session.cancelNum[0] for session in tradeRe]
				## feeList
				fee = [ computeFee(session.trades,openFee,closeFee,openRate,closeRate) for session in tradeRe]


				#### ~~~ append lists ~~~ ####

				tradeNumList.append(tradeNum)
				joiningTradeNumList.append(joiningTradeNum)
				improvingTradeNumList.append(improvingTradeNum)
				takingTradeNumList.append(takingTradeNum)
				cancelNumList.append(cancelNum)
				feeList.append(fee)

				## calculate terminalProfits and append 
				pre = 0.0
				terminalProfits = []
				for session in tradeRe:
					if len(session.profits):
						terminalProfits.append(session.profits[-1][1]) 
						pre= session.profits[-1][1]
					else:
						terminalProfits.append(pre)
				terminalProfitsList.append(terminalProfits)


			#### ~~~ save to attributes to pass back to StatsCompare ~~~ ####

			## tradeNumFrame
			self.tradeNumFrame = pd.DataFrame(tradeNumList).T
			self.tradeNumFrame.columns=['allSuccess','cancelFailed','takingFailed','improvingFailed','joinFailed','allFailed']			
			## joiningTradeNumFrame
			self.joiningTradeNumFrame = pd.DataFrame(joiningTradeNumList).T
			self.joiningTradeNumFrame.columns = self.tradeNumFrame.columns
			## improvingTradeNumFrame
			self.improvingTradeNumFrame = pd.DataFrame(improvingTradeNumList).T
			self.improvingTradeNumFrame.columns = self.tradeNumFrame.columns
			## takingTradeNumFrame
			self.takingTradeNumFrame = pd.DataFrame(takingTradeNumList).T
			self.takingTradeNumFrame.columns = self.tradeNumFrame.columns
			## feeFrame
			self.feeFrame = pd.DataFrame(feeList).T
			self.feeFrame.columns = self.tradeNumFrame.columns
			## cancelNumFrame
			self.cancelNumFrame = pd.DataFrame(cancelNumList).T
			self.cancelNumFrame.columns = self.tradeNumFrame.columns						
			## terminalFrame
			self.terminalFrame = pd.DataFrame(terminalProfitsList).T
			self.terminalFrame.columns = ['allSuccess','cancelFailed','takingFailed','improvingFailed','joinFailed','allFailed']
			self.terminalFrame['cancelDiff'] = self.terminalFrame.allSuccess - self.terminalFrame.cancelFailed
			self.terminalFrame['takingDiff'] = self.terminalFrame.allSuccess - self.terminalFrame.takingFailed
			self.terminalFrame['joinDiff'] = self.terminalFrame.allSuccess - self.terminalFrame.joinFailed
			self.terminalFrame['improvingDiff'] = self.terminalFrame.allSuccess - self.terminalFrame.improvingFailed
			self.terminalFrame['allDiff'] = self.terminalFrame.allSuccess - self.terminalFrame.allFailed  
			self.terminalFrame.fillna(method= 'ffill',inplace=True)
