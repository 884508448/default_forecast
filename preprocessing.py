import pandas as pd
import numpy as np
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

from utils.logger import LOGGER
from utils.consts import *
from utils.feature_encoder import encoder


# LC: loan Characteristics
# LP: Loan Periodic
# LCIS: Loan Characteristics Investment Status

class Preprocessing():
    def __init__(self) -> None:
        if os.path.exists(LC_DATA_PATH):
            LOGGER.info("loading lc_data ...")
            self.lc_data = pd.read_csv(LC_DATA_PATH)
            self.lc_data = self.lc_data.set_index("ListingId")
        else:
            LOGGER.info("generating lc_data ...")
            self.lc_data = pd.read_csv("data/LC.csv")
            # use ListingID as index of data
            self.lc_data = self.lc_data.set_index("ListingId")
            self.lc_data["借款成功日期"] = pd.to_datetime(self.lc_data["借款成功日期"])
            self.lc_data.to_csv(LC_DATA_PATH)
            self.lc_data  = self.lc_data

        if os.path.exists(LP_DATA_PATH):
            LOGGER.info("loading lp_data ...")
            self.lp_data = pd.read_csv(LP_DATA_PATH)
            self.lp_data = self.lp_data.set_index("ListingId")
        else:
            LOGGER.info("generating lp_data ...")
            self.lp_data = pd.read_csv("data/LP.csv")
            self.lp_data["到期日期"] = pd.to_datetime(self.lp_data["到期日期"])
            self.lp_data["还款日期"] = pd.to_datetime(self.lp_data["还款日期"].replace("\\N", "2017-1-31"))
            self.lp_data["recorddate"] = datetime.datetime(2017, 1, 31)
            # add new column: "逾期天数"
            self.lp_data["逾期天数"] = (self.lp_data["还款日期"] - self.lp_data["到期日期"]) / np.timedelta64(1, "D")
            self.lp_data["逾期天数"] = np.where(self.lp_data["逾期天数"] < 0, 0, self.lp_data["逾期天数"])
            self.lp_data = self.lp_data.set_index("ListingId")
            self.lp_data.to_csv(LP_DATA_PATH)
            self.lp_data = self.lp_data

        if os.path.exists(LCLP_DATA_PATH):
            LOGGER.info("loading lclp_data ...")
            self.lclp_data = pd.read_csv(LCLP_DATA_PATH)
            self.lclp_data = self.lclp_data.set_index("ListingId")
        else:
            LOGGER.info("generating lclp_data ...")
            # computing the number of repayments on unsetteled loans
            unsetteled_loans = (
                self.lp_data[(self.lp_data["还款状态"] != 3) & (self.lp_data["剩余利息"] > 0)]
                .groupby("ListingId")["期数"]
                .min()
                .reset_index()
            )
            # add  new column: "已还期数"
            unsetteled_loans['已还期数'] = unsetteled_loans['期数'] - 1
            unsetteled_loans = unsetteled_loans.set_index('ListingId')
            del unsetteled_loans['期数']

            self.lclp_data = pd.concat([self.lc_data, unsetteled_loans], axis=1, join='outer')
            self.lclp_data.loc[self.lclp_data['已还期数'].isnull(), '已还期数'] = self.lclp_data[self.lclp_data['已还期数'].isnull()]['借款期限']

            # computing overdue times
            self.lp_data['逾期天数'].replace(0, np.nan, inplace=True)
            delay_times = self.lp_data.groupby('ListingId')['逾期天数'].count()
            delay_times.name = '本笔已逾期次数'
            self.lp_data['逾期天数'].replace(np.nan, 0, inplace=True)
            self.lclp_data = pd.concat([self.lclp_data, delay_times], axis=1, join='outer').fillna(0)
            self.lclp_data.to_csv(LCLP_DATA_PATH)
            self.lclp_data = self.lclp_data

        
        LOGGER.info("generating loan_data ...")
        # group loans
        group_loan1 = self.lp_data.groupby('ListingId').agg({'剩余本金': 'sum', '剩余利息': 'sum', '还款状态': 'max'})
        group_loan2 = self.lp_data.drop(columns=self.lp_data.columns[1: 9]).groupby('ListingId').max()
        group_loan1.rename(columns={'剩余本金': '剩余未还本金', '剩余利息': '剩余未还利息'}, inplace=True)
        group_loan = pd.concat([group_loan1, group_loan2], axis=1)
        LOGGER.debug("group_loan info:")
        self.loan = pd.concat([self.lclp_data, group_loan], axis=1, join='outer')
        self.loan['历史逾期还款占比'] = (100 * self.loan['历史逾期还款期数']/(self.loan['历史逾期还款期数'] + self.loan['历史正常还款期数'])).round(2).fillna(0)
        self.loan['年龄段'] = pd.cut(self.loan['年龄'], bins=[15, 20, 25, 30, 35, 40, 45, 50, 60])
        self.loan['借款期限段'] = pd.cut(self.loan['借款期限'], bins=[3*i for i in range(9)])
        self.loan['target'] = np.where((self.loan['逾期天数'] > 60) & (self.loan['剩余未还利息'] > 0), 1, 0)
        self.loan["借款成功日期"] = pd.to_datetime(self.loan["借款成功日期"])
        self.loan['借款成功日期'] =(self.loan['借款成功日期']-pd.to_datetime('2015-01-01'))/ (np.timedelta64(1,'D'))

        # target 与逾期天数之间相关性很高，需要剔除
        del self.loan['逾期天数']
        # 特征编码
        self.loan = encoder(self.loan)
        self.loan.to_csv("data/loan.csv")


if __name__ == "__main__":
    processer = Preprocessing()







