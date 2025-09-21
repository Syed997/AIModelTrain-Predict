    def __getitem__(self, idx):
        x = self.data[idx:idx+self.input_window]
        y = self.data[idx+self.input_window:idx+self.input_window+self.forecast_horizon]
        return x, y