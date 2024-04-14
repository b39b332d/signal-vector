class KalmanFilter1D:
    def __init__(self,R =0.1 ** 2  ,Q = 1e-5):
        # intial guesses
        self.xhat = 0  # a posteri estimate of x
        self.P = 1.0  # a posteri error estimate


        self.R = R  # estimate of measurement variance
        self.Q = Q # process variance

        self.xhatminus = 0  # a priori estimate of x
        self.Pminus = 0  # a priori error estimate
        self.K = 0  # gain or blending factor
    def predict(self):
        self.xhatminus = self.xhat
        self.Pminus = self.P + self.Q
        return self.xhatminus

    def correct(self, sig):
        # measurement update
        self.K = self.Pminus / (self.Pminus + self.R)
        self.xhat = self.xhatminus + self.K * (sig - self.xhatminus)
        self.P = (1 - self.K) * self.Pminus

        return self.xhat

    def update(self, sig):
        self.predict()
        return self.correct(sig)

