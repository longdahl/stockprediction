import pickle
import calendar
import yfinance as yf
import yahoofinancials
import tensorflow as tf
config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1, 'CPU': 4})