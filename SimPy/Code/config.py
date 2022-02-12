#### ~~~~~~~~~~~~~~~~~~~~~~ ####
####     Configurations     ####
#### ~~~~~~~~~~~~~~~~~~~~~~ ####


## trading sessions
session_spans_rb_ = ( 	
					((21*60)*60*10**6,(23*60)*60*10**6),
					((9*60)*60*10**6,(10*60+15)*60*10**6),
					((10*60+30)*60*10**6,(11*60+30)*60*10**6),
					((13*60+30)*60*10**6,(15*60)*60*10**6)
					)
session_spans_rm_ = ( 	
					((21*60)*60*10**6,(23*60+30)*60*10**6),
					((9*60)*60*10**6,(10*60+15)*60*10**6),
					((10*60+30)*60*10**6,(11*60+30)*60*10**6),
					((13*60+30)*60*10**6,(15*60)*60*10**6)
					)
session_spans_ta_ = (
					((21*60)*60*10**6,(23*60+30)*60*10**6), 	
					((9*60)*60*10**6,(10*60+15)*60*10**6),
					((10*60+30)*60*10**6,(11*60+30)*60*10**6),
					((13*60+30)*60*10**6,(15*60)*60*10**6)
					)
session_spans_i_ = ( 	
					((21*60)*60*10**6,(23*60)*60*10**6),
					((9*60)*60*10**6,(10*60+15)*60*10**6),
					((10*60+30)*60*10**6,(11*60+30)*60*10**6),
					((13*60+30)*60*10**6,(15*60)*60*10**6)
					)	
session_spans_ng_ = ( 	
					((21*60)*60*10**6,(2*60+30)*60*10**6),
					((3*60)*60*10**6,(4*60+15)*60*10**6),
					((5*60+30)*60*10**6,(6*60+30)*60*10**6),
					((7*60+30)*60*10**6,(8*60)*60*10**6)
					)
session_spans_au_ = ( 	
					((21*60)*60*10**6,(2*60+30)*60*10**6),
					((9*60)*60*10**6,(10*60+15)*60*10**6),
					((10*60+30)*60*10**6,(11*60+30)*60*10**6),
					((13*60+30)*60*10**6,(15*60)*60*10**6)
					)
session_spans_sc_ = ( 	
					((21*60)*60*10**6,(2*60+30)*60*10**6),
					((9*60)*60*10**6,(10*60+15)*60*10**6),
					((10*60+30)*60*10**6,(11*60+30)*60*10**6),			
					((13*60+30)*60*10**6,(15*60)*60*10**6)
					)					
session_spans_ni_ = ( 	
					((21*60)*60*10**6,(1*60)*60*10**6),
					((9*60)*60*10**6,(10*60+15)*60*10**6),
					((10*60+30)*60*10**6,(11*60+30)*60*10**6),
					((13*60+30)*60*10**6,(15*60)*60*10**6)
					)

session_spans_ap_ = ( 					
					((9*60)*60*10**6,(10*60+15)*60*10**6),
					((10*60+30)*60*10**6,(11*60+30)*60*10**6),
					((13*60+30)*60*10**6,(15*60)*60*10**6)
					)
session_spans_sp_ = ( 	
					((21*60)*60*10**6,(23*60)*60*10**6),
					((9*60)*60*10**6,(10*60+15)*60*10**6),
					((10*60+30)*60*10**6,(11*60+30)*60*10**6),
					((13*60+30)*60*10**6,(15*60)*60*10**6)
					)
session_spans_ic_ = \
session_spans_ih_ = \
session_spans_if_ = (  	
					((9*60+29)*60*10**6,(11*60+30)*60*10**6),
					((13*60)*60*10**6,(15*60)*60*10**6)
					)


## statusGlobals
from collections import namedtuple
direction_ = namedtuple('direction', ['BUY','SELL'])(BUY=1, SELL=-1)
of_flag_ = namedtuple('of_flag',['OPEN','CLOSE','INIT'])(OPEN=1, CLOSE=2, INIT=3)
orderStatus_ = namedtuple('orderStatus',['LIVE','DEAD','CANCELED'])(LIVE=0, DEAD=1, CANCELED=2)
submitType_ = namedtuple('submitType',['JOINING','IMPROVING','TAKING'])(JOINING=0, IMPROVING=1, TAKING=2)
drivenType_ = namedtuple('drivenType',['TARGET','ASSIST','LIQUIDATION','ORDER'])(TARGET=0, ASSIST=1, LIQUIDATION=2, ORDER=3)
logNameType_ = namedtuple('logNameType',['SUBMIT','TRADE','CANCEL'])(SUBMIT=1, TRADE=2, CANCEL=3)


## volMultipleDict
volMultipleDict_ = {
	"IC0001": 200, 
	"IF0001": 300, 
	"IH0001": 300,
	"IC0002": 200, 
	"IF0002": 300, 
	"IH0002": 300,
	"RM0001": 10,
	"RM0002": 10,
	"M0001": 10,
	"M0002": 10,
	"M0003": 10,
	"AU0001": 1000,
	"NI0001": 1,
	"AP0001": 10,
	"RB0001": 10,	
	"RB0002": 10,
	"J0001": 100,
	"HC0001": 10,
	"SC0001": 1000,
	"FG0001": 20,
	"SP0001": 10,
	"MA0001": 10,
	"CU0001": 5,
	"I0001": 100,
	"I0002": 100,	
	"I0003": 100,
	"I0004": 100,
	"RU0001": 10,
	"BZ0001": 1000,
	"CL0001": 1000,
	"P0001": 10,
	"P0002": 10,
	"TA0001": 5,
	"TA0002": 5,
	"EG0001": 10,
	"J0001": 100,
	"JM0001": 60,
	"JD0001": 10,
	"JD0002": 10,
	"FU0001": 10,
	"FU0002": 10,
	"BU0001": 10,
	"BU0002": 10,
	"OI0001": 10,
	"CF0001": 5,
	"CJ0001": 5,
	"SM0001": 5,
	"SF0001": 5,
	"Y0001": 10,
	"PP0001": 5,
	"L0001": 5,
	"V0001": 5,
	"AGTD0001": 1,
	"GC0001": 100,
	"SI0001": 5000,
	"AG0001": 15,
	"AG0002": 15,
	"C0001": 10,
	"CS0001": 10,
	"ZN0001": 5,
	"ZN0002": 5,
	"PG0001": 20,
	"ZS0001": 5000,
	"FC0001": 25,
	"ZL0001": 27,	
	"AUTD0001": 1000,
	"ICUS_DX":1000,
}


## MINMOVE
minMoveDict_ = {
	"IC0001": 0.2, 
	"IF0001": 0.2, 
	"IH0001": 0.2,
	"IC0002": 0.2, 
	"IF0002": 0.2, 
	"IH0002": 0.2,	
	"RM0001": 1.0,
	"RM0002": 1.0,
	"M0001":  1.0,
	"M0002":  1.0,
	"M0003":  1.0,
	"AU0001": 0.02,
	"NI0001": 10.0,
	"AP0001": 1.0,
	"RB0001": 1.0,
	"RB0002": 1.0,
	"I0001": 0.5,
	"I0002": 0.5,
	"I0003": 0.5,
	"I0004": 0.5,
	"J0001": 0.5,
	"HC0001": 1.0,
	"SC0001": 0.1,
	"FG0001": 1.0,
	"SP0001": 2.0,
	"MA0001": 1.0,
	"CU0001": 10.0,		
	"RU0001": 5,
	"BZ0001": 0.01,
	"CL0001": 0.01,	
	"P0001": 2,
	"P0002": 2,
	"TA0001": 2,
	"TA0002": 2,
	"EG0001": 1,
	"J0001": 0.5,
	"JM0001": 0.5,
	"JD0001": 1,
	"JD0002": 1,
	"FU0001": 1,
	"FU0002": 1,
	"BU0001": 2,
	"BU0002": 2,
	"OI0001": 1,
	"CF0001": 5,
	"CJ0001": 5,
	"SM0001": 2,
	"SF0001": 2,
	"Y0001": 2,
	"PP0001": 1,
	"L0001": 5,
	"V0001": 5,
	"AGTD0001": 1,
	"GC0001": 0.01,
	"SI0001": 0.005,
	"AG0001": 1,
	"AG0002": 1,
	"C0001": 1,
	"CS0001": 1,
	"ZN0001": 5,
	"ZN0002": 5,
	"PG0001": 1,
	"ZS0001": 0.25,
	"FC0001": 1,
	"ZL0001": 0.01,
	"AUTD0001": 0.01,
	"ICUS_DX":0.005,
}

# L2 data source 
DataSourceDict_ = {
    "L2": [3,5,6,7,9,13,15,23,27,28,36,39,40,42],
    "FEX": [41],
    "OdrRtn" :[43],
}



