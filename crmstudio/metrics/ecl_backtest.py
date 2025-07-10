import numpy as np

def coverage_ratio(predicted_ecl, actual_loss):
    predicted_sum = np.sum(predicted_ecl)
    actual_sum = np.sum(actual_loss)
    if actual_sum == 0:
        return np.nan
    return predicted_sum / actual_sum

def bias(predicted_ecl, actual_loss):
    return np.sum(predicted_ecl) - np.sum(actual_loss)

def mae(predicted_ecl, actual_loss):
    return np.mean(np.abs(predicted_ecl - actual_loss))

def loss_emergence_over_time(actual_losses, time_periods):
    """
    Given actual_losses as a DataFrame indexed by time periods,
    compute cumulative loss emergence ratio over time.
    """
    cum_loss = actual_losses.cumsum()
    total_loss = actual_losses.sum()
    if total_loss == 0:
        return np.nan
    return cum_loss / total_loss
