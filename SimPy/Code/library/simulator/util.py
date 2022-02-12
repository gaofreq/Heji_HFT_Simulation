#### ~~~~~~~~~~~~~ ####
####   Libraries   ####
#### ~~~~~~~~~~~~~ ####

import numpy as np
from numba import njit
import pandas as pd
import datetime as dt
import math

from config import of_flag_



@njit
def clip(value, lower, upper):
	ret = value
	if lower > upper:
		return ret
	if ret < lower:
		ret = lower
	elif ret > upper:
		ret = upper

	return ret


#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~~ function: get_session() ~~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
"""
Get session. If outside trading time, 0 is assigned.

Parameters
---------------
exchange_time: ndarray of dtype np.int64 
session_spans: tuple, each entry is of the form (start,end),
			   where start and end are microseconds in a day.

Returns
----------------
session: ndarray. 

"""
@njit
def get_session(exchange_time, session_spans):
									 
	sessions = np.zeros_like(exchange_time)
	us_in_day = 86400000000      
 
	for i in range(len(exchange_time)):
		if exchange_time[i] > 100000000000000000:  # 18-digit timestamp, ns  
			current_time = (exchange_time[i]/1000)%us_in_day  # ns/1000 -> ms
		else:
			current_time = exchange_time[i]%us_in_day
		for j,span in enumerate(session_spans,1):
			start = span[0]
			end = span[1]
			
			if start< end:
				if current_time>= start and current_time < end:
					sessions[i] = j
					break
			else:
				if current_time>= start or current_time < end:
					sessions[i] = j
					break				
	return sessions



#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~~ function: datetimeFromUs() ~~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
'''
def datetimeFromUs(x): 
	if x is None:
		return pd.NaT
	if np.isscalar(x):
		if np.isnan(x):
			return pd.NaT
		## determine start of time
		time_1970 = dt.datetime(1970,1,1) + dt.timedelta(microseconds=int(x))
		time_2000 = dt.datetime(2000,1,1) + dt.timedelta(microseconds=int(x))
		time_now = dt.datetime.now()
		year_start = 1970 if abs((time_1970-time_now).total_seconds()) < abs((time_2000-time_now).total_seconds()) else 2000	
		return dt.datetime(year_start,1,1) + dt.timedelta(microseconds=int(x))
	if isinstance(x, (pd.Series,pd.DataFrame)):
		## determine start of time
		time_1970 = dt.datetime(1970,1,1) + pd.to_timedelta(x.tolist()[0]*1000)
		time_2000 = dt.datetime(2000,1,1) + pd.to_timedelta(x.tolist()[0]*1000)
		time_now = dt.datetime.now()
		year_start = 1970 if abs((time_1970-time_now).total_seconds()) < abs((time_2000-time_now).total_seconds()) else 2000			
		return pd.to_timedelta(x*1000) + dt.datetime(year_start,1,1)
'''
def datetimeFromUs(x): 
	if x is None:
		return pd.NaT
	if np.isscalar(x):
		if np.isnan(x):
			return pd.NaT
		## determine start of time
		time_1970 = dt.datetime(1970,1,1) + dt.timedelta(microseconds=int(x))
		time_2000 = dt.datetime(2000,1,1) + dt.timedelta(microseconds=int(x))
		time_now = dt.datetime.now()
		year_start = 1970 if abs((time_1970-time_now).total_seconds()) < abs((time_2000-time_now).total_seconds()) else 2000	
		return dt.datetime(year_start,1,1) + dt.timedelta(microseconds=int(x))
	if isinstance(x, (pd.Series,pd.DataFrame)):
		## determine start of time
		time_1970 = dt.datetime(1970,1,1) + pd.to_timedelta(x.tolist()[0])
		time_2000 = dt.datetime(2000,1,1) + pd.to_timedelta(x.tolist()[0])
		time_now = dt.datetime.now()
		year_start = 1970 if abs((time_1970-time_now).total_seconds()) < abs((time_2000-time_now).total_seconds()) else 2000
		return pd.to_timedelta(x) + dt.datetime(year_start,1,1)


#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~~ function: datetimeInUs() ~~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####

def datetimeInUs(x):
	if isinstance(x, dt.datetime):
		x = x - dt.datetime(2000,1,1)
		return int(x.microseconds + (x.seconds + x.days * 24 * 3600) * 10**6)
	if isinstance(x, (pd.Series,pd.DataFrame)):
		return ((x - dt.datetime(2000,1,1)).fillna(0).astype(np.int64)/1000+0.5).astype(np.int64)



#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~~ function: computeFee() ~~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####

def computeFee(trades,openFee,closeFee,openRate,closeRate):

	if len(trades):
		trades = pd.DataFrame(trades,columns=trades[0]._fields)
		return (((trades.of_flag==of_flag_.OPEN)*(openFee+trades.tradePrice*openRate)
				+(trades.of_flag == of_flag_.CLOSE)*(closeFee+trades.tradePrice*closeRate))*trades.volume).sum()
	else:
		return 0.0    



#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~~ function: combineOrders() ~~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####

def combineOrders(order_df, refId_list):
	order_df=order_df.copy()
	order_df["VOLUME"] = order_df.groupby(["REF_ID"])["VOLUME"].unique().sum()[0]
	order_df["CURRENT_VOLUME"] = sum([([1 if x==y else np.nan for y in order_df["REF_ID"]]*order_df["CURRENT_VOLUME"]).fillna(method='ffill').fillna(method='bfill') for x in refId_list])
	order_df["REF_ID"] = min(refId_list)
	order_df["STATUS"] = [1]*(order_df.shape[0]-1)+[0]
	return(order_df)




#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~~ function: updateBidStack() ~~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####

@njit
def updateBidStack(bidStack, bidStackNoSpoofing, currentMkt, signals, params, MINMOVE):
	bidIndex = int(round((currentMkt.bid-signals.bidBottomPrice)/MINMOVE))
	bidStackCapacity = len(bidStack)
	preTop = bidStack[signals.bidLen-1]
	preTopNoSproofing = bidStackNoSpoofing[signals.bidLen-1]

	if bidIndex<=0:
		if currentMkt.L2==0.0:
			bidStack[0] = currentMkt.bz
			bidStackNoSpoofing[0] = min(currentMkt.bz, params.spoofingLevel)
			signals.bidLen = 1
			signals.bidTopPrice = currentMkt.bid
			signals.bidBottomPrice = currentMkt.bid
		if currentMkt.L2==1.0:
			bidStack[0] = currentMkt.bz4
			bidStackNoSpoofing[0] = min(currentMkt.bz4, params.spoofingLevel)
			bidStack[1] = bidStack[0] + currentMkt.bz3
			bidStackNoSpoofing[1] = bidStackNoSpoofing[0] + min(currentMkt.bz3, params.spoofingLevel)
			bidStack[2] = bidStack[1] + currentMkt.bz2
			bidStackNoSpoofing[2] = bidStackNoSpoofing[1] + min(currentMkt.bz2, params.spoofingLevel)
			bidStack[3] = bidStack[2] + currentMkt.bz1
			bidStackNoSpoofing[3] = bidStackNoSpoofing[2] + min(currentMkt.bz1, params.spoofingLevel)
			bidStack[4] = bidStack[3] + currentMkt.bz
			bidStackNoSpoofing[4] = bidStackNoSpoofing[3] + min(currentMkt.bz, params.spoofingLevel)
			signals.bidLen = 5
			signals.bidTopPrice = currentMkt.bid
			signals.bidBottomPrice = currentMkt.bid4 if currentMkt.bid4>0.0 else currentMkt.bid-10*MINMOVE 				
	elif bidIndex < signals.bidLen:
		if currentMkt.L2==0.0:
			bidStack[bidIndex] = currentMkt.bz + bidStack[bidIndex-1]
			bidStackNoSpoofing[bidIndex] = min(currentMkt.bz,params.spoofingLevel) + bidStackNoSpoofing[bidIndex-1]
			signals.bidLen = bidIndex+1
			signals.bidTopPrice = currentMkt.bid
		if currentMkt.L2==1.0:
			if bidIndex > 4:
				bidStack[bidIndex-4] = bidStack[bidIndex-5] + currentMkt.bz4
				bidStack[bidIndex-3] = bidStack[bidIndex-4] + currentMkt.bz3
				bidStack[bidIndex-2] = bidStack[bidIndex-3] + currentMkt.bz2
				bidStack[bidIndex-1] = bidStack[bidIndex-2] + currentMkt.bz1
				bidStack[bidIndex] = bidStack[bidIndex-1] + currentMkt.bz
				bidStackNoSpoofing[bidIndex-4] = bidStackNoSpoofing[bidIndex-5] + min(currentMkt.bz4, params.spoofingLevel)
				bidStackNoSpoofing[bidIndex-3] = bidStackNoSpoofing[bidIndex-4] + min(currentMkt.bz3, params.spoofingLevel)
				bidStackNoSpoofing[bidIndex-2] = bidStackNoSpoofing[bidIndex-3] + min(currentMkt.bz2, params.spoofingLevel)
				bidStackNoSpoofing[bidIndex-1] = bidStackNoSpoofing[bidIndex-2] + min(currentMkt.bz1, params.spoofingLevel)
				bidStackNoSpoofing[bidIndex] = bidStackNoSpoofing[bidIndex-1] + min(currentMkt.bz, params.spoofingLevel)
				signals.bidLen = bidIndex+1
				signals.bidTopPrice = currentMkt.bid
			else:
				bidStack[0] = currentMkt.bz4
				bidStackNoSpoofing[0] = min(currentMkt.bz4, params.spoofingLevel)
				bidStack[1] = bidStack[0] + currentMkt.bz3
				bidStackNoSpoofing[1] = bidStackNoSpoofing[0] + min(currentMkt.bz3, params.spoofingLevel)
				bidStack[2] = bidStack[1] + currentMkt.bz2
				bidStackNoSpoofing[2] = bidStackNoSpoofing[1] + min(currentMkt.bz2, params.spoofingLevel)
				bidStack[3] = bidStack[2] + currentMkt.bz1
				bidStackNoSpoofing[3] = bidStackNoSpoofing[2] + min(currentMkt.bz1, params.spoofingLevel)
				bidStack[4] = bidStack[3] + currentMkt.bz
				bidStackNoSpoofing[4] = bidStackNoSpoofing[3] + min(currentMkt.bz, params.spoofingLevel)
				signals.bidLen = 5
				signals.bidTopPrice = currentMkt.bid
				signals.bidBottomPrice = currentMkt.bid4 if currentMkt.bid4>0.0 else currentMkt.bid-10*MINMOVE				
	elif bidIndex < bidStackCapacity:
		if currentMkt.L2==0.0:
			for j in range(signals.bidLen,bidIndex):
				bidStack[j] = preTop
				bidStackNoSpoofing[j] = preTopNoSproofing
			bidStack[bidIndex] = currentMkt.bz + preTop
			bidStackNoSpoofing[bidIndex] = min(currentMkt.bz,params.spoofingLevel) + preTopNoSproofing
			signals.bidLen = bidIndex+1
			signals.bidTopPrice = currentMkt.bid
		if currentMkt.L2==1.0:
			if bidIndex > 4:
				if signals.bidLen <= bidIndex-4:
					for j in range(signals.bidLen,bidIndex-4):
						bidStack[j] = preTop
						bidStackNoSpoofing[j] = preTopNoSproofing
				bidStack[bidIndex-4] = bidStack[bidIndex-5] + currentMkt.bz4
				bidStack[bidIndex-3] = bidStack[bidIndex-4] + currentMkt.bz3
				bidStack[bidIndex-2] = bidStack[bidIndex-3] + currentMkt.bz2
				bidStack[bidIndex-1] = bidStack[bidIndex-2] + currentMkt.bz1
				bidStack[bidIndex] = bidStack[bidIndex-1] + currentMkt.bz
				bidStackNoSpoofing[bidIndex-4] = bidStackNoSpoofing[bidIndex-5] + min(currentMkt.bz4, params.spoofingLevel)
				bidStackNoSpoofing[bidIndex-3] = bidStackNoSpoofing[bidIndex-4] + min(currentMkt.bz3, params.spoofingLevel)
				bidStackNoSpoofing[bidIndex-2] = bidStackNoSpoofing[bidIndex-3] + min(currentMkt.bz2, params.spoofingLevel)
				bidStackNoSpoofing[bidIndex-1] = bidStackNoSpoofing[bidIndex-2] + min(currentMkt.bz1, params.spoofingLevel)
				bidStackNoSpoofing[bidIndex] = bidStackNoSpoofing[bidIndex-1] + min(currentMkt.bz, params.spoofingLevel)
				signals.bidLen = bidIndex+1
				signals.bidTopPrice = currentMkt.bid
			else:
				bidStack[0] = currentMkt.bz4
				bidStackNoSpoofing[0] = min(currentMkt.bz4, params.spoofingLevel)
				bidStack[1] = bidStack[0] + currentMkt.bz3
				bidStackNoSpoofing[1] = bidStackNoSpoofing[0] + min(currentMkt.bz3, params.spoofingLevel)
				bidStack[2] = bidStack[1] + currentMkt.bz2
				bidStackNoSpoofing[2] = bidStackNoSpoofing[1] + min(currentMkt.bz2, params.spoofingLevel)
				bidStack[3] = bidStack[2] + currentMkt.bz1
				bidStackNoSpoofing[3] = bidStackNoSpoofing[2] + min(currentMkt.bz1, params.spoofingLevel)
				bidStack[4] = bidStack[3] + currentMkt.bz
				bidStackNoSpoofing[4] = bidStackNoSpoofing[3] + min(currentMkt.bz, params.spoofingLevel)
				signals.bidLen = 5
				signals.bidTopPrice = currentMkt.bid
				signals.bidBottomPrice = currentMkt.bid4 if currentMkt.bid4>0.0 else currentMkt.bid-10*MINMOVE
	else:
		##this case is rare if the initial length of bidStack is sufficiently large
		print("bidStack is Full!")




#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~~ function: updateAskStack() ~~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####

@njit
def updateAskStack(askStack, askStackNoSpoofing, currentMkt, signals, params, MINMOVE):
	askIndex = int(round((-currentMkt.ask+signals.askBottomPrice)/MINMOVE))
	askStackCapacity = len(askStack)
	preTop = askStack[signals.askLen-1]
	preTopNoSproofing = askStackNoSpoofing[signals.askLen-1]

	if askIndex<=0:
		if currentMkt.L2==0.0:
			askStack[0] = currentMkt.az
			askStackNoSpoofing[0] = min(currentMkt.az, params.spoofingLevel)
			signals.askLen = 1
			signals.askTopPrice = currentMkt.ask
			signals.askBottomPrice = currentMkt.ask
		if currentMkt.L2==1.0:
			askStack[0] = currentMkt.az4
			askStackNoSpoofing[0] = min(currentMkt.az4, params.spoofingLevel)
			askStack[1] = askStack[0] + currentMkt.az3
			askStackNoSpoofing[1] = askStackNoSpoofing[0] + min(currentMkt.az3, params.spoofingLevel)
			askStack[2] = askStack[1] + currentMkt.az2
			askStackNoSpoofing[2] = askStackNoSpoofing[1] + min(currentMkt.az2, params.spoofingLevel)
			askStack[3] = askStack[2] + currentMkt.az1
			askStackNoSpoofing[3] = askStackNoSpoofing[2] + min(currentMkt.az1, params.spoofingLevel)
			askStack[4] = askStack[3] + currentMkt.az
			askStackNoSpoofing[4] = askStackNoSpoofing[3] + min(currentMkt.az, params.spoofingLevel)
			signals.askLen = 5
			signals.askTopPrice = currentMkt.ask
			signals.askBottomPrice = currentMkt.ask4 if currentMkt.ask4>0 else currentMkt.ask+10*MINMOVE
	elif askIndex < signals.askLen:
		if currentMkt.L2==0.0:
			askStack[askIndex] = currentMkt.az + askStack[askIndex-1]
			askStackNoSpoofing[askIndex] = min(currentMkt.az, params.spoofingLevel) + askStackNoSpoofing[askIndex-1]
			signals.askLen = askIndex+1
			signals.askTopPrice = currentMkt.ask
		if currentMkt.L2==1.0:
			if askIndex > 4:
				askStack[askIndex-4] = askStack[askIndex-5] + currentMkt.az4
				askStack[askIndex-3] = askStack[askIndex-4] + currentMkt.az3
				askStack[askIndex-2] = askStack[askIndex-3] + currentMkt.az2
				askStack[askIndex-1] = askStack[askIndex-2] + currentMkt.az1
				askStack[askIndex] = askStack[askIndex-1] + currentMkt.az
				askStackNoSpoofing[askIndex-4] = askStackNoSpoofing[askIndex-5] + min(currentMkt.az4, params.spoofingLevel)
				askStackNoSpoofing[askIndex-3] = askStackNoSpoofing[askIndex-4] + min(currentMkt.az3, params.spoofingLevel)
				askStackNoSpoofing[askIndex-2] = askStackNoSpoofing[askIndex-3] + min(currentMkt.az2, params.spoofingLevel)
				askStackNoSpoofing[askIndex-1] = askStackNoSpoofing[askIndex-2] + min(currentMkt.az1, params.spoofingLevel)
				askStackNoSpoofing[askIndex] = askStackNoSpoofing[askIndex-1] + min(currentMkt.az, params.spoofingLevel)
				signals.askLen = askIndex+1
				signals.askTopPrice = currentMkt.ask
			else:
				askStack[0] = currentMkt.az4
				askStackNoSpoofing[0] = min(currentMkt.az4, params.spoofingLevel)
				askStack[1] = askStack[0] + currentMkt.az3
				askStackNoSpoofing[1] = askStackNoSpoofing[0] + min(currentMkt.az3, params.spoofingLevel)
				askStack[2] = askStack[1] + currentMkt.az2
				askStackNoSpoofing[2] = askStackNoSpoofing[1] + min(currentMkt.az2, params.spoofingLevel)
				askStack[3] = askStack[2] + currentMkt.az1
				askStackNoSpoofing[3] = askStackNoSpoofing[2] + min(currentMkt.az1, params.spoofingLevel)
				askStack[4] = askStack[3] + currentMkt.az
				askStackNoSpoofing[4] = askStackNoSpoofing[3] + min(currentMkt.az, params.spoofingLevel)
				signals.askLen = 5
				signals.askTopPrice = currentMkt.ask
				signals.askBottomPrice = currentMkt.ask4 if currentMkt.ask4>0 else currentMkt.ask+10*MINMOVE
	elif askIndex < askStackCapacity:
		if currentMkt.L2==0.0:
			for j in range(signals.askLen,askIndex):
				askStack[j] = preTop
				askStackNoSpoofing[j] = preTopNoSproofing
			askStack[askIndex] = currentMkt.az + preTop
			askStackNoSpoofing[askIndex] = min(currentMkt.az,params.spoofingLevel) + preTopNoSproofing
			signals.askLen = askIndex+1
			signals.askTopPrice = currentMkt.ask
		if currentMkt.L2==1.0:
			if askIndex > 4:
				if signals.askLen <= askIndex-4:
					for j in range(signals.askLen,askIndex-4):
						askStack[j] = preTop
						askStackNoSpoofing[j] = preTopNoSproofing
				askStack[askIndex-4] = askStack[askIndex-5] + currentMkt.az4
				askStack[askIndex-3] = askStack[askIndex-4] + currentMkt.az3
				askStack[askIndex-2] = askStack[askIndex-3] + currentMkt.az2
				askStack[askIndex-1] = askStack[askIndex-2] + currentMkt.az1
				askStack[askIndex] = askStack[askIndex-1] + currentMkt.az
				askStackNoSpoofing[askIndex-4] = askStackNoSpoofing[askIndex-5] + min(currentMkt.az4, params.spoofingLevel)
				askStackNoSpoofing[askIndex-3] = askStackNoSpoofing[askIndex-4] + min(currentMkt.az3, params.spoofingLevel)
				askStackNoSpoofing[askIndex-2] = askStackNoSpoofing[askIndex-3] + min(currentMkt.az2, params.spoofingLevel)
				askStackNoSpoofing[askIndex-1] = askStackNoSpoofing[askIndex-2] + min(currentMkt.az1, params.spoofingLevel)
				askStackNoSpoofing[askIndex] = askStackNoSpoofing[askIndex-1] + min(currentMkt.az, params.spoofingLevel)
				signals.askLen = askIndex+1
				signals.askTopPrice = currentMkt.ask				
			else:
				askStack[0] = currentMkt.az4
				askStackNoSpoofing[0] = min(currentMkt.az4, params.spoofingLevel)
				askStack[1] = askStack[0] + currentMkt.az3
				askStackNoSpoofing[1] = askStackNoSpoofing[0] + min(currentMkt.az3, params.spoofingLevel)
				askStack[2] = askStack[1] + currentMkt.az2
				askStackNoSpoofing[2] = askStackNoSpoofing[1] + min(currentMkt.az2, params.spoofingLevel)
				askStack[3] = askStack[2] + currentMkt.az1
				askStackNoSpoofing[3] = askStackNoSpoofing[2] + min(currentMkt.az1, params.spoofingLevel)
				askStack[4] = askStack[3] + currentMkt.az
				askStackNoSpoofing[4] = askStackNoSpoofing[3] + min(currentMkt.az, params.spoofingLevel)
				signals.askLen = 5
				signals.askTopPrice = currentMkt.ask
				signals.askBottomPrice = currentMkt.ask4 if currentMkt.ask4>0 else currentMkt.ask+10*MINMOVE				
	else:
		##this case is rare if the initial length of askStack is sufficiently large
		print("askStack is Full!")
        
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~~ function:updateOrderBookByNoTrade() ~~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####

@njit
def updateOrderBookByNoTrade(direction, limitPrice, submitVolume, leftVolume,signals,bidBook,askBook,MINMOVE):

	if (limitPrice < signals.topBid - 10 * MINMOVE) or (limitPrice > signals.topAsk + 10 * MINMOVE):
		return False
	if leftVolume == 0:
		return False
	# buy orders
	if direction == 0 : # BUY
		if signals.bidBookIsEmpty == True:
			signals.topBid = limitPrice
			signals.bottomBid = limitPrice
			signals.topBidIdx = 0

			bidBook[0] = submitVolume
			signals.bidBookIsEmpty = False

			return True

		bidIdx = math.floor((limitPrice - signals.bottomBid)/MINMOVE + 0.5)

		if (limitPrice <= signals.topBid) and (limitPrice >= signals.bottomBid):
			bidBook[bidIdx] = bidBook[bidIdx] + submitVolume
		elif limitPrice > signals.topBid:
			if limitPrice >= signals.topAsk:
				askIdx = math.floor((signals.bottomAsk - limitPrice)/MINMOVE + 0.5)
				for j in range(askIdx,signals.topAskIdx+1):                # delete ask < limitPrice 
					askBook[j] = 0
                  
				signals.topAskIdx = askIdx - 1
				signals.topAsk = limitPrice + MINMOVE                 
				while ((askBook[signals.topAskIdx] == 0)&(signals.topAskIdx>=0)):    # to find non-zero ask price
					signals.topAskIdx -= 1
					signals.topAsk += MINMOVE
					if signals.topAskIdx < -1:
						break                    
                       
				if signals.topAsk > signals.bottomAsk:
					signals.askBookIsEmpty = True
					signals.topAsk = 99999999.0
					signals.bottomAsk = 99999999.0
					signals.topAskIdx = 99999999
              
			for j in range(signals.topBidIdx+1,bidIdx):     # fill zeros between topBid and limitPrice
				bidBook[j] = 0

			bidBook[bidIdx] = submitVolume
			signals.topBid = limitPrice
			signals.topBidIdx = bidIdx
	return True
'''             
		elif limitPrice < signals.bottomBid:

			moveIdx = - bidIdx

			for j in range(0,signals.topBidIdx+1):
				bidBook[signals.topBidIdx+moveIdx-j] = bidBook[signals.topBidIdx-j]
			signals.bottomBid = limitPrice
			bidBook[0] = submitVolume

			for j in range(1,moveIdx):
				bidBook[j] = 0

			signals.topBidIdx += moveIdx

	elif direction == 1:
		if signals.askBookIsEmpty == True:
			signals.topAsk = limitPrice
			signals.bottomAsk = limitPrice
			signals.topAskIdx = 0

			askBook[0] = submitVolume
			signals.askBookIsEmpty = False

			return True

		askIdx = math.floor((signals.bottomAsk - limitPrice) / MINMOVE + 0.5)

		if (limitPrice <= signals.bottomAsk) & (limitPrice >= signals.topAsk):
			askBook[askIdx] = askBook[askIdx] + submitVolume

		elif limitPrice < signals.topAsk:
			if limitPrice <= signals.topBid:
				bidIdx = math.floor((limitPrice - signals.bottomBid) / MINMOVE + 0.5)

				for j in range(bidIdx,signals.topBidIdx+1):
					bidBook[j] = 0

				signals.topBidIdx = bidIdx -1
				signals.topBid = limitPrice - MINMOVE

				while ((bidBook[signals.topBidIdx] == 0) & (signals.topBidIdx >= 0)):
					print(signals.topBidIdx,"bid")
					signals.topBidIdx -= 1
					signals.topBid -= MINMOVE
					if signals.topBidIdx < -1:
						break 

				if signals.topBid < signals.bottomBid:
					signals.bidBookIsEmpty = True
					signals.topBid = 0.0
					signals.bottomBid = 0.0
					signals.topBidIdx = 0

			for j in range(signals.topAskIdx+1,askIdx):
				askBook[j] = 0

			askBook[askIdx] = submitVolume
			signals.topAsk = limitPrice
			signals.topAskIdx = askIdx

		elif limitPrice > signals.bottomAsk:
			moveIdx = -askIdx

			for j in range(0,signals.topAskIdx+1):
				askBook[signals.topAskIdx + moveIdx + j] = askBook[signals.topAskIdx+j] 
			signals.bottomAsk = limitPrice
			askBook[0] = submitVolume

			for j in (1,moveIdx):
				askBook[j] = 0

			signals.topAskIdx += moveIdx

	return True
'''
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####
#### ~~~~ function:updateOrderBookByTrade() ~~~~ ####
#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####

@njit
def updateOrderBookByTrade(direction, limitPrice, submitVolume, tradedVolume, leftVolume,isSelfSubmit,isBeforeNextMkt,signals,bidBook,askBook,MINMOVE):

	signals.preLastPrice = limitPrice
    
	# buy orders
	if direction == 0 : # BUY

		if signals.bidBookIsEmpty == True:
			if leftVolume == 0:
				return True                
			else:
				signals.topBid = limitPrice
				signals.bottomBid = limitPrice
				signals.topBidIdx = 0

				bidBook[0] = leftVolume
				signals.bidBookIsEmpty = False

				return True

		bidIdx = math.floor((limitPrice - signals.bottomBid)/MINMOVE + 0.5)

		if (limitPrice <= signals.topBid) and (limitPrice >= signals.bottomBid):
			for j in range(bidIdx+1,signals.topBidIdx+1):                # delete bid > limitPrice             
				bidBook[j] = 0
                
			signals.topBid = limitPrice
			signals.topBidIdx = bidIdx

			if isSelfSubmit and isBeforeNextMkt:
				if leftVolume == 0:
					signals.topBidIdx -= 1
					signals.topBid -= MINMOVE
					while ((bidBook[signals.topBidIdx] == 0)&(signals.topBidIdx >= 0)):  # find non-zero bid
						signals.topBidIdx -= 1
						signals.topBid -= MINMOVE
					if signals.topBid < signals.bottomBid:
						signals.bidBookIsEmpty = True
						signals.topBid = 0.0
						signals.bottomBid = 0.0
						signals.topBidIdx = 0
				else:
					bidBook[bidIdx] = leftVolume

			else:
				if bidBook[bidIdx] > tradedVolume:
					bidBook[bidIdx] = max(bidBook[bidIdx]-tradedVolume,leftVolume)
				else:
					if leftVolume > 0:
						bidBook[bidIdx] = leftVolume
					else:
						bidBook[bidIdx] = 0
						signals.topBidIdx -= 1
						signals.topBid -= MINMOVE

						while ((bidBook[signals.topBidIdx]==0) &(signals.topBidIdx >= 0)):
							signals.topBidIdx -= 1
							signals.topBid -= MINMOVE
						if signals.topBid < signals.bottomBid:
							signals.bidBookIsEmpty = True
							signals.topBid = 0.0
							signals.bottomBid = 0.0
							signals.topBidIdx = 0
                            
		elif (limitPrice > signals.topBid) and (limitPrice<signals.topAsk):
			if leftVolume > 0 :
				for j in range(signals.topBidIdx,bidIdx):
					bidBook[bidIdx] = 0
				bidBook[bidIdx] = leftVolume
				signals.topBid = limitPrice
				signals.topBidIdx = bidIdx

		elif (limitPrice >= signals.topAsk):
			signals.preLastPrice = signals.topAsk
                
			if tradedVolume < askBook[signals.topAskIdx]:
				askBook[signals.topAskIdx] = askBook[signals.topAskIdx] - tradedVolume
			else:
				if signals.askBookIsEmpty == True:    
					return True
				signals.topAskIdx -= 1
				signals.topAsk += MINMOVE

				while ((askBook[signals.topAskIdx]==0) and (signals.topAskIdx>=0)):
					signals.topAskIdx -= 1
					signals.topAsk += MINMOVE


				if (signals.topAsk > signals.bottomAsk):
					signals.askBookIsEmpty = True
					signals.topAsk = 99999999.0
					signals.bottomAsk = 99999999.0
					signals.topAskIdx = 99999999



				if isSelfSubmit:
					return False

		else:
			for j in range(0,signals.topBidIdx+1):
				bidBook[j] = 0
			if leftVolume > 0:
				signals.topBid = limitPrice
				signals.topBidIdx = 0
				signals.bottomBid = limitPrice
				bidBook[0] = leftVolume
			else:
				signals.bidBookIsEmpty = True
				signals.topBid = 0.0
				signals.topBidIdx = 0
				signals.bottomBid = 0
                

	# sell orders
	elif direction == 1 : # SELL
		if signals.askBookIsEmpty == True:
			if leftVolume == 0:
				return True                
			else:
				signals.topAsk = limitPrice
				signals.bottomAsk = limitPrice
				signals.topAskIdx = 0

				askBook[0] = leftVolume
				signals.askBookIsEmpty = False

				return True

		askIdx = math.floor((signals.bottomAsk - limitPrice)/MINMOVE + 0.5)

		if (limitPrice >= signals.topAsk) and (limitPrice <= signals.bottomAsk):
			for j in range(askIdx+1,signals.topAskIdx+1):                # delete ask < limitPrice             
				askBook[j] = 0
                
			signals.topAsk = limitPrice
			signals.topAskIdx = askIdx

			if isSelfSubmit and isBeforeNextMkt:
				if leftVolume == 0:
					signals.topAskIdx -= 1
					signals.topAsk -= MINMOVE
					while ((askBook[signals.topAskIdx]==0)&(signals.topAskIdx >= 0)):  # find non-zero ask
						signals.topAskIdx -= 1
						signals.topAsk += MINMOVE
					if signals.topAsk > signals.bottomAsk:
						signals.askBookIsEmpty = True
						signals.topAsk = 99999999.0
						signals.bottomAsk = 99999999.0
						signals.topAskIdx = 99999999
				else:
					askBook[askIdx] = leftVolume

			else:
				if askBook[askIdx] > tradedVolume:
					askBook[askIdx] = max(askBook[askIdx]-tradedVolume,leftVolume)
				else:
					if leftVolume > 0:
						askBook[askIdx] = leftVolume
					else:
						askBook[askIdx] = 0
						signals.topAskIdx -= 1
						signals.topAsk += MINMOVE

						while ((askBook[signals.topAskIdx] == 0)&(signals.topAskIdx >= 0)):
							signals.topAskIdx -= 1
							signals.topAsk += MINMOVE
						if signals.topAsk < signals.bottomAsk:
							signals.askBookIsEmpty = True
							signals.topAsk = 99999999.0
							signals.bottomAsk = 99999999.0
							signals.topAskIdx = 99999999

		elif (limitPrice > signals.topBid) and (limitPrice<signals.topAsk):
			if leftVolume > 0 :
				for j in range(signals.topAskIdx,askIdx):
					askBook[askIdx] = 0
				askBook[askIdx] = leftVolume
				signals.topAsk = limitPrice
				signals.topAskIdx = askIdx

		elif (limitPrice <= signals.topBid):
			signals.preLastPrice = signals.topBid #############
			if tradedVolume < bidBook[signals.topBidIdx]:
				bidBook[signals.topBidIdx] = bidBook[signals.topBidIdx] - tradedVolume
			else:
				signals.topBidIdx -= 1
				signals.topBid += MINMOVE

				while ((bidBook[signals.topBidIdx]==0) and (signals.topBidIdx>=0)):
					signals.topBidIdx -= 1
					signals.topBid -= MINMOVE #########

				if (signals.topBid < signals.bottomBid) or (signals.topBidIdx < 0):
					signals.bidBookIsEmpty = True
					signals.topBid = 99999999.0
					signals.bottomBid = 99999999.0
					signals.topBidIdx = 99999999

				if isSelfSubmit:
					return False

		else:
			for j in range(0,signals.topAskIdx+1):
				askBook[j] = 0
			if leftVolume > 0:
				signals.topAsk = limitPrice
				signals.topAskIdx = 0
				signals.bottomAsk = limitPrice
				askBook[0] = leftVolume
			else:
				signals.askBookIsEmpty = True
				signals.topAsk = 99999999.0
				signals.topAskIdx = 99999999
				signals.bottomAsk = 99999999.0
                
	return True               

                
                
