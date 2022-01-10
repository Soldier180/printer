from logging import DEBUG, basicConfig, getLogger
from random import randint, random

import matplotlib.pyplot as plt


class KalmanFilter(object):
    def __init__(
        self,
        initial_estimate: float = random(),
        initial_est_error: float = random(),
        initial_measure_error: float = random(),
    ):
        self.estimate = initial_estimate
        self.gain = random()
        self.est_error = initial_est_error
        self.measure_error = initial_measure_error
        self.sensor_values = []


    def calculate_kalman_gain(self) -> None:
        """calculates Kalman gain given error values"""
        self.gain = self.est_error / (self.est_error + self.measure_error)

    def update_estimate(self, sensor_value: int = 0.0) -> None:
        """updates estimate based on Kalman gain"""
        new_estimate = self.estimate + self.gain * (sensor_value - self.estimate)
        self.estimate = new_estimate

    def calculate_estimate_error(self) -> None:
        """calculates error of the updated estimate"""
        self.est_error = (1 - self.gain) * self.est_error

    def iterative_updates(self, v) -> None:
        e = []
        self.calculate_kalman_gain()
        self.update_estimate(sensor_value=v)
        self.calculate_estimate_error()




if __name__ == "__main__":
    kf = KalmanFilter(
        initial_estimate=68.0,
        initial_est_error=2.0,
        initial_measure_error=4.0,
    )
    kf.iterative_updates()