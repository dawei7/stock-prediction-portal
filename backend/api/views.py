from django.shortcuts import render
from rest_framework.views import APIView
from .serializers import StockPredictionSerializer
from rest_framework import status
from rest_framework.response import Response

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import os

from django.conf import settings
from .utils import save_plot

from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score


class StockPredictionAPIView(APIView):
    def post(self, request):
        serializer = StockPredictionSerializer(data=request.data)
        if serializer.is_valid():
            ticker = serializer.validated_data['ticker']

            # -----------------------------
            # 1) Fetch Data from Yahoo Finance
            # -----------------------------
            now = datetime.now()
            start = datetime(now.year - 10, now.month, now.day)  # 10 years
            end = now
            df = yf.download(ticker, start, end)
            if df.empty:
                return Response(
                    {"error": "No data found for the given ticker.",
                     'status': status.HTTP_404_NOT_FOUND}
                )

            df = df.reset_index()  # so that 'Date' is a column
            # Convert date to Timestamp (good practice for plotting, etc.)
            df['Date'] = pd.to_datetime(df['Date'])

            # -----------------------------
            # 2) Basic Plots
            # -----------------------------
            plt.switch_backend('AGG')

            # 2.1) Closing Price
            plt.figure(figsize=(12, 5))
            plt.plot(df['Close'], label='Closing Price')
            plt.title(f'Closing price of {ticker}')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.legend()
            plot_img_path = f'{ticker}_plot.png'
            plot_img = save_plot(plot_img_path)

            # 2.2) 100-Day MA
            ma100 = df['Close'].rolling(100).mean()
            plt.switch_backend('AGG')
            plt.figure(figsize=(12, 5))
            plt.plot(df['Close'], label='Closing Price')
            plt.plot(ma100, 'r', label='100 DMA')
            plt.title(f'100 Days Moving Average of {ticker}')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.legend()
            plot_100_path = f'{ticker}_100_dma.png'
            plot_100_dma = save_plot(plot_100_path)

            # 2.3) 200-Day MA
            ma200 = df['Close'].rolling(200).mean()
            plt.switch_backend('AGG')
            plt.figure(figsize=(12, 5))
            plt.plot(df['Close'], label='Closing Price')
            plt.plot(ma100, 'r', label='100 DMA')
            plt.plot(ma200, 'g', label='200 DMA')
            plt.title(f'200 Days Moving Average of {ticker}')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.legend()
            plot_200_path = f'{ticker}_200_dma.png'
            plot_200_dma = save_plot(plot_200_path)

            # -----------------------------
            # 3) Train/Test Split
            # -----------------------------
            # 70% train, 30% test
            data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
            data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.7):])

            # -----------------------------
            # 4) Scale and Prepare Model
            # -----------------------------
            scaler = MinMaxScaler(feature_range=(0, 1))

            # Load your trained model
            model_path = os.path.join(settings.BASE_DIR, "resources", "stock_prediction_model.keras")
            model = load_model(model_path)

            # -----------------------------
            # 5) Prepare Test Data
            # -----------------------------
            # We'll combine the last 100 days of training + all test data
            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

            # Scale it
            input_data = scaler.fit_transform(final_df)

            x_test = []
            y_test = []
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i - 100:i])
                y_test.append(input_data[i, 0])  # The next day's close

            x_test, y_test = np.array(x_test), np.array(y_test)

            # Predict on test set
            y_predicted = model.predict(x_test)

            # -----------------------------
            # 6) Inverse-Transform Predictions
            # -----------------------------
            y_predicted = scaler.inverse_transform(y_predicted).flatten()
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

            # -----------------------------
            # 7) Plot Predictions vs. Actual (Test)
            # -----------------------------
            plt.switch_backend('AGG')
            plt.figure(figsize=(12, 5))
            plt.plot(y_test, 'b', label='Original Price')
            plt.plot(y_predicted, 'r', label='Predicted Price')
            plt.title(f'Final Prediction on Test Set ({ticker})')
            plt.xlabel('Test Data Points')
            plt.ylabel('Price')
            plt.legend()
            final_pred_path = f'{ticker}_final_prediction.png'
            plot_prediction = save_plot(final_pred_path)

            # -----------------------------
            # 8) Model Evaluation
            # -----------------------------
            mse = mean_squared_error(y_test, y_predicted)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_predicted)

            # -----------------------------
            # 9) **Predict +1 Day** Beyond Our Dataset
            # -----------------------------
            # We will take the last 100 days from the ENTIRE df (not just the test set)
            # because we want to predict the day after the dataset ends
            last_100_full = df['Close'].tail(100).values.reshape(-1, 1)

            # NOTE: We must use the *same* scaler that was fit on final_df,
            # so let's keep that 'scaler'.
            # However, that scaler was fit on final_df. If you want to
            # scale the entire dataset, you may want to re-fit on all data.
            # For simplicity, we'll reuse the same `scaler` as final_df:
            #  (In practice, you'd want to be consistent with your training approach.)
            scaled_100 = scaler.transform(last_100_full)

            X_final = []
            X_final.append(scaled_100)
            X_final = np.array(X_final)  # shape: (1, 100, 1)

            next_pred = model.predict(X_final)  # shape: (1, 1)
            next_pred = scaler.inverse_transform(next_pred).flatten()[0]  # single float
            next_pred_price = float(np.round(next_pred, 2))

            # The date after the last row in df
            last_real_date = df['Date'].iloc[-1]
            future_date = last_real_date + timedelta(days=1)

            # -----------------------------
            # 10) Plot +1 Day Prediction
            # -----------------------------
            # We can do a simple plot that shows the test set predictions + new day
            plt.switch_backend('AGG')
            plt.figure(figsize=(12, 5))
            plt.plot(y_test, 'b', label='Original Price (Test)')      # Blue line for test
            plt.plot(y_predicted, 'r', label='Predicted Price (Test)')  # Red line for predicted

            # We'll place the next day point at index len(y_test)
            plt.scatter(len(y_test), next_pred_price,
                        color='green',
                        label=f'Next Day Prediction\n{future_date.date()}: {next_pred_price}')
            plt.title(f'Prediction for Next Day Beyond Dataset ({ticker})')
            plt.xlabel('Test Data Points + 1 Future')
            plt.ylabel('Price')
            plt.legend()

            # Save the “+1 day” plot
            plus_one_path = f'{ticker}_plus_one_prediction.png'
            plus_one_plot = save_plot(plus_one_path)

            # -----------------------------
            # 11) Return Response
            # -----------------------------
            return Response({
                'status': 'success',
                'ticker': ticker,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,

                'last_close_in_dataset': float(df['Close'].iloc[-1]),
                'predicted_next_day_price': next_pred_price,
                'future_date': str(future_date.date()),

                # Plots
                'plot_img': plot_img,
                'plot_100_dma': plot_100_dma,
                'plot_200_dma': plot_200_dma,
                'plot_prediction': plot_prediction,
                'plot_plus_one': plus_one_plot
            })
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
