from modeltester import ModelTester

mt = ModelTester()
mt.loadData("train.csv")
data = mt.data
data.head()

mt.plotCorrelation()
