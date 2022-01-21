

if __name__ == "__main__":
    kf = KalmanFilter(
        initial_estimate=68.0,
        initial_est_error=2.0,
        initial_measure_error=4.0,
    )
    kf.iterative_updates()