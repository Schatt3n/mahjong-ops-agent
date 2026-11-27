"""统一模拟店家的营业窗口、跨日开场和散场后的重订边界。"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta


@dataclass(frozen=True, slots=True)
class VenueHours:
    """营业配置按当地时刻循环。相同开关门时刻表示连续营业。"""

    opens: time
    closes: time

    @classmethod
    def from_config(cls, venue: dict) -> "VenueHours":
        """读取同一份模拟门店配置。"""
        return cls(time.fromisoformat(venue["opens"]), time.fromisoformat(venue["closes"]))

    def service_window(self, at: datetime) -> tuple[datetime, datetime | None]:
        """找覆盖当前或接下来的连续营业窗口，全天营业没有关门上界。"""
        if self.opens == self.closes:
            return at, None
        day = timedelta(days=1)
        overnight = self.closes < self.opens
        business_date = at.date() - day if overnight and at.time() < self.closes else at.date()
        opens = datetime.combine(business_date, self.opens, tzinfo=at.tzinfo)
        closes = datetime.combine(business_date + day if overnight else business_date, self.closes, tzinfo=at.tzinfo)
        return (opens + day, closes + day) if at >= closes else (opens, closes)

    def covers(self, start: datetime, end: datetime | None) -> bool:
        """开场须在营业中，已知结束时刻允许正好等于关门。"""
        opens, closes = self.service_window(start)
        return opens <= start and (end is None or start < end and (closes is None or end <= closes))
