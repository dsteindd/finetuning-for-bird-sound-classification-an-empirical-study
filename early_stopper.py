class EarlyStopping:

    def __init__(self,
                 patience=5,
                 mode="min"
                 ):
        self.current_best = 10000 if mode == "min" else -10000
        self.mode = mode
        self.patience = patience

        self.current_p = 0

    def update(self, value):
        if self.mode == "min":
            if value < self.current_best:
                self.current_best = value
                self.current_p = 0
            else:
                self.current_p += 1
        else:
            if value > self.current_best:
                self.current_best = value
                self.current_p = 0
            else:
                self.current_p += 1

    def stop_criterion_reached(self):
        return self.current_p >= self.patience
