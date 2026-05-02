import pandas as pd

# Global modeling configurations
# This date is used to separate training data (for profiling, clustering and fitting)
# from testing data (for evaluation). 
TEST_CUTOFF = "2011-09-01"
TEST_CUTOFF_DT = pd.to_datetime(TEST_CUTOFF)
