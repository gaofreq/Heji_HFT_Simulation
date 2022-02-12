from collections import namedtuple
import numpy as np


params = namedtuple('params', [  
	## common params   
	'fracCut',
	'queueEdgeAlpha',	
	'alpha',
	'initVar',
	'varEwmaDecay',
	'varScaler',
	'theoGamma',
	'theoAlpha',      
	'theoGamma2',
	'theoAlpha2',  
	'volumeEwmaDecay',
	'vol2SizeRatioLowerBound1',
	'spoofingLevel',
	'spreadEwmaDecay',
	'spreadScaler',
	'qI',
	'minimumEdge',
	'minimumEdgeRate',
	'timeToClose',
	'posPenCoef',
	'posPenIntercept',
	'posPenLowerBound',
	'posPenLowerBoundCancel',
	'penaltyPower',
	'signalFloor',
	'quoteRange',
	'riskQuoteRange',
	'maxQuoteSize',
	'maxUnitOfRisk',
	'cmeScaler',
	'cancelScalerOnAssist',
	'minEdge2',
	'improveTick',
	'spreadLowerBound',
	'spreadUpperBound',

	## for mktMakingCZCE only  
	'varBound',
	'spreadCoef',
	'detQuoteSize',
	'detVolShock',
	'detFillRatio',
	'maxResQuoteSize',
	'maxNorUnitOfRisk',
	'maxResUnitOfRisk',
	'oneSidedMaxPos',
	'minSnapshotVolumeToImpAndTake',
	'fracCutImprove',
	'fracCutTake',
	'joinOnlyPrevLevels',
	'useSigmaAdjust',

	## for kronos_base only  
	'qFracMethod',
	'fractCutImprove',
	'improveQI',
	'improveQI1',
	'improveQueueBeta',
	'signalVarScaler',
	'takeQI',
	'takeQI1',
	'takeQueueBeta',
	'takingEdgeCond',
	'requirePredCrossMid',
	'varAlpha4J',
	'varBeta4J',
	'varBase4J',
	'varAlpha4I',
	'varBeta4I',
	'varBase4I',	
	'varAlpha4T',
	'varBeta4T',
	'varBase4T', 
	'volumeEwmaDecay2',
	'qBeta',
	'qGamma',
	'qFacMethod',
	'I',
	'J1',
	'J2',
	'K',
	'maxNumAtBest',
	'sizeEwmaDecay',
	'useExt',
	'minStandizedSignalForTaking',
	'minSnapshotVolumeTake',
	'volBound4Joining',
	'volBound4Taking',
	'minSnapshotVolumForTaking',
	'maxSnapshotVolumeImp',
	'volBound4Improving',
	'joinOnlyOnOldLevels',
	'qAlpha',
	'vol2SizeRatioLowerBound0',
	'useNewCancel',
	'probCancel',
	'useOrderDriven',

	## other params 
	'extL2',
	'fractCutTake',
	'minEdge0',
	'tfactor',
	'varCut',
	'detAbsCut',
	'detFracCut',
	'maxDetOrderOnTop',
	'riskCancelRange',
	'flexibleUnitOfRisk',
	'useOutSideSignal',
    'SZALLThreshold',
    'SZALLbeta',
    'signalScoreB',
    'signalScoreC',    

	## assist orders related
	'theoReturnOpenAssist1',
	'notActivatedAssistInsRoot1',
	'notActivatedAssistInsRoot2',
	'notActivatedAssistInsRoot3',
	'notActivatedAssistInsRoot4',
	'notActivatedAssistInsRoot5',
	'activatedAssistInsRoot',
	'activatedAssistInsRoot2',    
	'useAssistReturn',
	'assistReturnBeta',
	'useIncreAssistTheo',
	'increAssistTheoThreshold',    
	'useIncreAssistSignal',
	'increAssistSignalThreshold',
    
	'orderReturnBeta',
    'assistReturnType',
    
	## factor placeholders for reserach
	'G_Factor1',
	'G_Factor2',
	'G_Factor3',
	'L_Factor1',
	'L_Factor2',
	'L_Factor3',
	'Y_Factor1',
	'Y_Factor2',
	'Y_Factor3',
	'Y_Factor4',
	'Y_Factor5',
	'Y_Factor6',
	'Y_Factor7',
	'Y_Factor8',	

	])



paramsInstance = params(
	## common params
	fracCut = 0.25,
	queueEdgeAlpha= 1.0,	
	alpha=1.0,
	initVar = 0.2*0.2,
	varEwmaDecay=(1.0 - 2.0/(200.0 + 1.0)),
	varScaler= 0.0,
	theoGamma = 1.0,   
	theoAlpha = 0.0,
	theoGamma2 = 1.0,
	theoAlpha2 = 0.0,

	volumeEwmaDecay = 0.9,
	vol2SizeRatioLowerBound1 = -1000.0,
	spoofingLevel = 10000.0,
	spreadEwmaDecay = 0.9,
	spreadScaler = 0.0,
	qI = 0.0,
	minimumEdge = 0.0,
	minimumEdgeRate = 0.0,
	timeToClose = np.timedelta64(60000000000, 'ns'),  ## 1 min
	posPenCoef = 10,
	posPenIntercept = 1.0,
	posPenLowerBound = 0.5,

	posPenLowerBoundCancel = 0.1,   ## only in simulation
	penaltyPower = 1.0,    ## only in simulation

	signalFloor = -1000.0,
	quoteRange=4,
	riskQuoteRange = 1,
	maxQuoteSize = 0,
	maxUnitOfRisk = 0,
	cmeScaler = 1.0, 
	cancelScalerOnAssist=0.0,
	minEdge2 = 1000.0,  ## only in simulation
	improveTick = 1.0,
	spreadLowerBound =  0.0,   ## only in simulation
	spreadUpperBound = 10000.0,  ## only in simulation

	## for mktMakingCZCE only  
	varBound = 0.0,
	spreadCoef = 0.0,
	detQuoteSize = 10000,
	detVolShock = 10000,
	detFillRatio = 1.5,
	maxResQuoteSize = 0,
	maxNorUnitOfRisk = 0,
	maxResUnitOfRisk = 0,
	oneSidedMaxPos = 0,
	minSnapshotVolumeToImpAndTake = -100,
	fracCutImprove = 0.25,
	fracCutTake = 0.25,
	joinOnlyPrevLevels = False,
	useSigmaAdjust = True,

	## for kronos_base only
	qFracMethod = 0,   
	fractCutImprove = 1.0,
	improveQI = 10.0,       ## not used
	improveQI1 = 100000.0,  ## not used
	improveQueueBeta = 0.0,   ## only in simulation
	signalVarScaler = 0.0,    ## only in simulation
	takeQI = 10.0,       ## not used
	takeQI1 = 100000.0,  ## not used
	takeQueueBeta = 0.0,   ## not used
	takingEdgeCond = 2,    ## not used
	requirePredCrossMid = False,
	varAlpha4J = 0.0,
	varBeta4J = 0.0,
	varBase4J = 1.0,
	varAlpha4I = 0.0,
	varBeta4I = 0.0,
	varBase4I = 1.0,  	
	varAlpha4T = 0.0,
	varBeta4T = 0.0,
	varBase4T = 1.0,
	volumeEwmaDecay2 = 0.1,
	qBeta = 0.5,    ## only in simulation
	qGamma = 0.0,   ## only in simulation
	qFacMethod = 0,  ## only in simulation
	I = 0.0,   ## only in simulation
	J1= 0.0,   ## only in simulation
	J2= 0.0,   ## only in simulation
	K = 0.0,   ## only in simulation
	maxNumAtBest = 1000,    ## only in simulation
	sizeEwmaDecay = 0.99,   ## only in simulation
	useExt = False,     ## only in simulation
	minStandizedSignalForTaking = -100.0,
	minSnapshotVolumeTake = -1.0,
	volBound4Joining = 10000,     ## only in simulation
	volBound4Taking = -1000.0,    ## only in simulation
	minSnapshotVolumForTaking = -1.0,
	maxSnapshotVolumeImp = 100000.0,  ## only in simulation
	volBound4Improving = 10000,   ## only in simulation
	joinOnlyOnOldLevels = False,  ## only in simulation
	qAlpha = 0.5,   ## only in simulation
	vol2SizeRatioLowerBound0 = 0,   ## only in simulation
	useNewCancel = False,
	probCancel = 1e-50,
	useOrderDriven = False,    

	## other params   
	extL2 = False,
	fractCutTake = 1.0,  
	minEdge0 = -100,
	tfactor = 1.0,
	varCut = 1000,
	detAbsCut = 0,
   	detFracCut = 0,
   	maxDetOrderOnTop = 0,
	riskCancelRange = -1,
	flexibleUnitOfRisk = 0,	                   
	useOutSideSignal = False,
    SZALLThreshold = 0.0,
    SZALLbeta = 0.0,
    signalScoreB = 0.0,
    signalScoreC = 1000,
    
	## assist orders related
	theoReturnOpenAssist1 = -1,
	notActivatedAssistInsRoot1 = -1,
	notActivatedAssistInsRoot2 = -1,
	notActivatedAssistInsRoot3= -1,
	notActivatedAssistInsRoot4 = -1,
	notActivatedAssistInsRoot5 = -1,
	activatedAssistInsRoot = -1,
	activatedAssistInsRoot2 = -1,
	useAssistReturn = False,
	assistReturnBeta = 0,
	useIncreAssistTheo = False,
	increAssistTheoThreshold = 0,
	useIncreAssistSignal = False,
	increAssistSignalThreshold = 0,

	orderReturnBeta = 0,
    assistReturnType = 0,

	## factor placeholders for reserach
	G_Factor1 = 0.0,
	G_Factor2 = 0.0,
	G_Factor3 = 0.0,
	L_Factor1 = 0.0,
	L_Factor2 = 0.0,
	L_Factor3 = 0.0,
	Y_Factor1 = 0.0,
	Y_Factor2 = 0.0,
	Y_Factor3 = 0.0,
	Y_Factor4 = 0.0,
	Y_Factor5 = 0.0,
	Y_Factor6 = 0.0,
	Y_Factor7 = 0.0,
	Y_Factor8 = 0.0,

	)



## signals 
SignalsDtype = np.dtype([  ('numTicks',np.int32),
						   ('assistInitialized',np.bool),
						   ('mainThreadInitialized',np.bool),

						   ('bidLen', np.int32), 
						   ('askLen', np.int32),
						   ('bidTopPrice', np.double),
						   ('askTopPrice',np.double),
						   ('bidBottomPrice',np.double),
						   ('askBottomPrice',np.double),

						   ('bidQueueAfter', np.double),
						   ('askQueueAfter', np.double),
						   ('bidQueueFrac', np.double),
						   ('askQueueFrac', np.double),

						   ('squareMean',np.double),
						   ('mean',np.double),
						   ('varEwmaAdj',np.double),
						   ('var', np.double),
						   ('sigma', np.double),

						   ('theo', np.double),
						   ('theoLag', np.double),
						   ('currentSignal', np.double),

						   ('tradeImbal',np.double),
						   ('orderImbal',np.double),
						   ('orderImbalAbs',np.double),
						   ('orderImbalEwmaAdj',np.double),
						   ('tradeImbalEwmaAdj', np.double),

						   ('bidVolume',np.double),
						   ('askVolume',np.double),
						   ('buyVolume', np.double),
						   ('sellVolume', np.double),
						   ('buyVolumeShock', np.double),
						   ('sellVolumeShock', np.double),

						   ('volume1',np.double),
						   ('volume2',np.double),
						   ('ewmaSpread',np.double),
						   ('ewmaMid',np.double),
						   ('preBid',np.double),
						   ('preAsk',np.double),

						   ('preBz', np.double),
						   ('preAz', np.double),
						   ('bidSize',np.double),
						   ('askSize',np.double),
						   ('preLastPrice',np.double),
                         
						   ('orderBid',np.double),
						   ('orderBid1',np.double),
						   ('orderBid2',np.double),
						   ('orderBid3',np.double),
						   ('orderBid4',np.double),

						   ('orderBz',np.double),
						   ('orderBz1',np.double),
						   ('orderBz2',np.double),
						   ('orderBz3',np.double),
						   ('orderBz4',np.double),
                         
						   ('orderAsk',np.double),
						   ('orderAsk1',np.double),
						   ('orderAsk2',np.double),
						   ('orderAsk3',np.double),
						   ('orderAsk4',np.double),

						   ('orderAz',np.double),
						   ('orderAz1',np.double),
						   ('orderAz2',np.double),
						   ('orderAz3',np.double),
						   ('orderAz4',np.double),

						   ('moveAwayCount',np.int64),

						   ('minimumEdge', np.double),

						   ('bidMean',np.double),
						   ('bidSquareMean',np.double),
						   ('askMean',np.double),
						   ('askSquareMean',np.double),
						   ('volatility', np.double),
						   ('preBidSz',np.double),
						   ('preAskSz',np.double),
						   ('effectiveSpread',np.double),
						   ('effectiveEwmaAdj',np.double),
						   ('prePred',np.double),
						   ('ewmaTheo',np.double),
						   ('ewmaLastPrice', np.double),

						   ('estimatedPrevAccTunover', np.double),

						   ('firstAssistTheoAfterTarget', np.double), 
						   ('firstAssistFlag', np.bool), 
						   ('lastAssistTheoBeforeTarget', np.double), 
						   ('tmpLastAssistTheoBeforeTarget', np.double), 

                         
						   ('assistTheoLag', np.double),
						   ('ewmaAssistRet', np.double),

						   ('bzAll', np.double),
						   ('azAll', np.double),
						   ])