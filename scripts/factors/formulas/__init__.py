"""Pure price-volume formula functions, mapped by stable formula ID."""

from .activity import amount_relative_prior_5d, volume_relative_prior_5d
from .price import close_position_in_range, intraday_return, range_over_open
from .returns import close_return_1d, close_return_5d, close_return_20d


FORMULAS = {
    "returns.close_return_1d": close_return_1d,
    "returns.close_return_5d": close_return_5d,
    "returns.close_return_20d": close_return_20d,
    "price.intraday_return": intraday_return,
    "price.range_over_open": range_over_open,
    "price.close_position_in_range": close_position_in_range,
    "activity.volume_relative_prior_5d": volume_relative_prior_5d,
    "activity.amount_relative_prior_5d": amount_relative_prior_5d,
}
