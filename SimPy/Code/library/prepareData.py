#### ~~~ libraries ~~~ ####
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm

from library.simulator.util import get_session, datetimeFromUs, combineOrders
from config import direction_, logNameType_

#### ~~~ main logic ~~~ ###
def prepareData(sim_datafile_full_path, session_spans, doubleVolumeFlag, **kwargs):	

	#### ~~~ read in raw datafile ~~~ ####
	rawData = pd.read_csv(sim_datafile_full_path)

	#### ~~~ get optional arguments ~~~ ####
	instrumentRootOverwrite = kwargs.get('instrumentRootOverwrite', None)
	order_log_full_path = kwargs.get('order_log_full_path', None)


	## ~~~~~~~~~~~~~~~~~~~~~~~~ ##
	##     Prepare Mkt Data     ##
	## ~~~~~~~~~~~~~~~~~~~~~~~~ ##

	print("Preparing market data...")

	#### ~~~ clean data ~~~ ####
	rawData.sort_values('MachineTime', inplace=True)
	rawData.dropna(inplace=True)
	rawData = rawData.loc[(rawData.BidSize>0.5)&(rawData.AskSize>0.5)]
	rawData['trade_session'] = get_session(rawData.ExchangeTime.values, session_spans)
	rawData['session'] = rawData['trade_session']

	rawData['spread'] = (rawData['Ask'] - rawData['Bid']) * 1.0
	rawData = rawData.loc[(rawData.session > 0) & (rawData.spread > 0), :]
	rawData['mkt_date'] = rawData['MktDate'].map(str)
	rawData['session'] = rawData['mkt_date'].map(str)+' Session '+rawData['trade_session'].map(str)
	rawData['trade_session'] = rawData['trade_session'].astype('<f8')

	rawData['exchange_time'] = datetimeFromUs(rawData.ExchangeTime)
	rawData['machine_time'] = datetimeFromUs(rawData.MachineTime)

	rawData['instrument_root'] = rawData['Instrument'] if instrumentRootOverwrite is None else instrumentRootOverwrite
	rawData['source'] = 0.0

	rawData['bid'] = rawData['Bid'].astype('<f8')
	rawData['ask'] = rawData['Ask'].astype('<f8')
	rawData['bz'] = rawData['BidSize'].astype('<f8')
	rawData['az'] = rawData['AskSize'].astype('<f8')
	rawData['last_price'] = rawData['LastPrice'].astype('<f8')

	if "DATA_SOURCE" in rawData.columns:
		rawData['L2'] = rawData["DATA_SOURCE"].str.contains('L2', case=False) * 1.0
		rawData['bid1'] = rawData['Bid1'].astype('<f8')
		rawData['ask1'] = rawData['Ask1'].astype('<f8')
		rawData['bz1'] = rawData['BidSize1'].astype('<f8')
		rawData['az1'] = rawData['AskSize1'].astype('<f8')
		rawData['bid2'] = rawData['Bid2'].astype('<f8')
		rawData['ask2'] = rawData['Ask2'].astype('<f8')
		rawData['bz2'] = rawData['BidSize2'].astype('<f8')
		rawData['az2'] = rawData['AskSize2'].astype('<f8')
		rawData['bid3'] = rawData['Bid3'].astype('<f8')
		rawData['ask3'] = rawData['Ask3'].astype('<f8')
		rawData['bz3'] = rawData['BidSize3'].astype('<f8')
		rawData['az3'] = rawData['AskSize3'].astype('<f8')
		rawData['bid4'] = rawData['Bid4'].astype('<f8')
		rawData['ask4'] = rawData['Ask4'].astype('<f8')
		rawData['bz4'] = rawData['BidSize4'].astype('<f8')
		rawData['az4'] = rawData['AskSize4'].astype('<f8')
	else:
		rawData['L2'] = 0.0
		rawData['bid1'] = 0.0
		rawData['ask1'] = 0.0
		rawData['bz1'] = 0.0
		rawData['az1'] = 0.0
		rawData['bid2'] = 0.0
		rawData['ask2'] = 0.0
		rawData['bz2'] = 0.0
		rawData['az2'] = 0.0
		rawData['bid3'] = 0.0
		rawData['ask3'] = 0.0
		rawData['bz3'] = 0.0
		rawData['az3'] = 0.0
		rawData['bid4'] = 0.0
		rawData['ask4'] = 0.0
		rawData['bz4'] = 0.0
		rawData['az4'] = 0.0	
	rawData['bzAll'] = rawData['bz']+rawData['bz1']+rawData['bz2']+rawData['bz3']+rawData['bz4']
	rawData['azAll'] = rawData['az']+rawData['az1']+rawData['az2']+rawData['az3']+rawData['az4']

	if "AS_OF_DATE" in rawData.columns:
		rawData['AS_OF_DATE'] = rawData['AS_OF_DATE'].astype('<f8')
	else:
		rawData['AS_OF_DATE'] = [x.replace('-','') for x in rawData['mkt_date']]
		rawData['AS_OF_DATE'] = rawData['AS_OF_DATE'].astype('<f8')

	PredColnames = [x for x in rawData.columns if 'pred' in x.lower()]   ## prediction columns to be kept

	#### ~~~ calculate volume and turnover ~~~ ####
	rawData['AccVolume'] = rawData['AccVolume'].astype('<f8')
	rawData['AccTurnOver'] = rawData['AccTurnOver'].astype('<f8')
	rawData['volume'] = rawData.groupby(['instrument_root','session'])['AccVolume'].diff().fillna(0) 
	rawData['turnover'] = rawData.groupby(['instrument_root','session'])['AccTurnOver'].diff().fillna(0)

	if (rawData['volume']<0).sum()>0 :
		rawData['volume'] = rawData['Volume'].astype('<f8') 
		rawData['turnover'] = rawData['TurnOver'].astype('<f8') 		

	if doubleVolumeFlag:
		rawData['volume'] = rawData['volume'] / 2
		rawData['turnover'] = rawData['turnover'] / 2
		rawData['AccVolume'] = rawData['AccVolume'] / 2
		rawData['AccTurnOver'] = rawData['AccTurnOver'] / 2

	#### ~~~ subset to final data ~~~ ####
	output_colnames = ['az','az1','az2','az3','az4','bz','bz1','bz2','bz3','bz4',
					   'bid','bid1','bid2','bid3','bid4','ask','ask1','ask2','ask3','ask4','bzAll','azAll',
					   'instrument_root', 'turnover','volume', 'exchange_time','machine_time', 'spread', 'mkt_date',
					   'last_price','source','session','trade_session','L2', 'AccVolume', 'AccTurnOver', 'AS_OF_DATE']+PredColnames
	output_colnames = [x for x in output_colnames if x in rawData.columns]
	rawData = rawData[output_colnames]
	rawData.sort_values(by=["machine_time"], inplace=True)

	#### ~~~ save out to h5 file ~~~ ####
	h5_filename = sim_datafile_full_path.replace('csv', 'h5')
	ts_name = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))
	rawData.to_hdf(h5_filename, 'MKT_DATA_'+ts_name, mode='w')


	## ~~~~~~~~~~~~~~~~~~~~~~~~ ##
	##     Prepare Log Data     ##
	## ~~~~~~~~~~~~~~~~~~~~~~~~ ##

	#### ~~~ order log ~~~ ####
	if order_log_full_path is not None:

		print("Preparing order log data...")
		orderLog = pd.read_csv(order_log_full_path)
		orderLog.loc[orderLog["LOG_NAME"]=="submitting order","LOCAL_ORDER_ACK_TIME"] = orderLog.loc[orderLog["LOG_NAME"]=="submitting order","MACHINE_TIMESTAMP_DATA"]
		orderLog.loc[orderLog["LOG_NAME"]=="canceling order","LOCAL_ORDER_ACK_TIME"] = orderLog.loc[orderLog["LOG_NAME"]=="canceling order","TIME"]

		df_order = orderLog[["REF_ID","LOG_NAME","LOCAL_ORDER_ACK_TIME","VOLUME","CURRENT_VOLUME","DIRECTION","LIMIT_PRICE","STATUS","AS_OF_DATE"]].copy()
		df_order.rename(columns={"LOCAL_ORDER_ACK_TIME":"MACHINE_TIMESTAMP"}, inplace=True)
		df_order["DIRECTION"] = df_order["DIRECTION"].map(direction_._asdict())
		df_order["STATUS"] = df_order["STATUS"].apply(lambda x: 0 if x in ["DEAD","CANCELLED"] else 1)
		df_order["MACHINE_TIMESTAMP"] = pd.to_datetime(df_order["MACHINE_TIMESTAMP"])
		df_order["REF_ID"] = df_order["REF_ID"].str.replace('-','').astype(np.int64)

		df_order_cancel = df_order.loc[df_order["LOG_NAME"]=="canceling order",:].copy()
		df_order_cancel["LOG_NAME"] = "CANCEL"
		df_order_cancel["VOLUME"] = -1000
		df_order_cancel["CURRENT_VOLUME"] = -1000
		df_order_cancel.sort_values(by=["MACHINE_TIMESTAMP"], inplace=True)
		df_order_cancel = pd.merge_asof(df_order_cancel, rawData[["machine_time"]], left_on=["MACHINE_TIMESTAMP"], right_on=["machine_time"], direction="backward")
		df_order_cancel["MACHINE_TIMESTAMP"] = df_order_cancel["machine_time"]
		df_order_cancel.drop(columns=["machine_time"], inplace=True)
		df_order_cancel.sort_values(by=["AS_OF_DATE","REF_ID","MACHINE_TIMESTAMP"], inplace=True)

		df_order_insert = df_order.loc[df_order["LOG_NAME"]=="successful insert, updating live order",["AS_OF_DATE","REF_ID","MACHINE_TIMESTAMP"]].copy()
		df_order_insert.rename(columns={"MACHINE_TIMESTAMP":"INSERT_MACHINE_TIMESTAMP"}, inplace=True)

		df_order.drop_duplicates(subset=["AS_OF_DATE","REF_ID","CURRENT_VOLUME","STATUS"], inplace=True)
		df_order.sort_values(by=["MACHINE_TIMESTAMP"], inplace=True)

		## delete ifany invalid refIds
		temp = (df_order.groupby(["AS_OF_DATE","REF_ID"])["MACHINE_TIMESTAMP"].first()<df_order.groupby(["AS_OF_DATE","REF_ID"])["MACHINE_TIMESTAMP"].last())
		date_refId_list_delete = temp[temp==False].index
		if len(date_refId_list_delete)>0:
			print("The following REF_ID are deleted due to invalid submit time:")
			print(date_refId_list_delete.values)
			df_merge = pd.merge(df_order, temp[temp==False].reset_index().rename(columns={"MACHINE_TIMESTAMP":"VALID"}), on=["AS_OF_DATE","REF_ID"], how="left")
			df_order = df_merge.loc[df_merge["VALID"]!=False,:].copy()
			df_order.drop(columns=["VALID"],inplace=True)

		## combine orders submitted at the same time
		print("Combining orders...")
		temp = df_order.groupby(["AS_OF_DATE","MACHINE_TIMESTAMP","DIRECTION","LIMIT_PRICE"])["REF_ID"].count()
		to_combine = temp[temp>1].reset_index()[["AS_OF_DATE","MACHINE_TIMESTAMP","DIRECTION","LIMIT_PRICE"]]
		for i,r in tqdm(to_combine.iterrows(), ncols=69, total=to_combine.shape[0]):
			refId_list = df_order.loc[(df_order["MACHINE_TIMESTAMP"]==r["MACHINE_TIMESTAMP"])&(df_order["DIRECTION"]==r["DIRECTION"])&(df_order["LIMIT_PRICE"]==r["LIMIT_PRICE"]),"REF_ID"].unique().tolist()
			if len(refId_list)==0:
				continue
			df_order.loc[(df_order["REF_ID"].isin(refId_list))&(df_order["AS_OF_DATE"]==r["AS_OF_DATE"]),:] = combineOrders(df_order.loc[(df_order["REF_ID"].isin(refId_list))&(df_order["AS_OF_DATE"]==r["AS_OF_DATE"]),:], refId_list)
		df_order.drop_duplicates(subset=["AS_OF_DATE","REF_ID","CURRENT_VOLUME","STATUS"], inplace=True)

		## add traded volume 
		df_order["TRADED_VOLUME"] = -df_order.groupby(["AS_OF_DATE","REF_ID"])["CURRENT_VOLUME"].diff().fillna(-0.0)
		df_order_cancel["TRADED_VOLUME"] = 0

		## add cancel messages
		df_order["LOG_NAME"] = df_order.apply(lambda x: "SUBMIT" if x["VOLUME"]==x["CURRENT_VOLUME"] and x["STATUS"]==1 else "TRADE", axis=1)
		df_order = pd.concat([df_order, df_order_cancel], sort=False)
		df_order.sort_values(by=["MACHINE_TIMESTAMP"], inplace=True)
		df_order.reset_index(drop=True, inplace=True)
		df_order["LOG_NAME"] = df_order["LOG_NAME"].map(logNameType_._asdict())
		df_order["USED"] = 0
		df_order = pd.merge(df_order, df_order_insert, on=["AS_OF_DATE","REF_ID"], how="left")
		df_order.drop(columns=["AS_OF_DATE"],inplace=True)
		df_order = df_order.loc[df_order["INSERT_MACHINE_TIMESTAMP"].isnull()==False,:].copy()

		#### ~~~ save out to h5 file ~~~ ####
		df_order.to_hdf(h5_filename, 'ORDER_LOG_'+ts_name, mode='r+')



	## ~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
	##     save out to h5 file     ##
	## ~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##	
	ans = pd.HDFStore(h5_filename)


	#### ~~~ return ~~~ ####
	return(ans)












