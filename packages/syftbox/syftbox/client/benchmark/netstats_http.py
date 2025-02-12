import time
from dataclasses import dataclass
from io import BytesIO

from curl_cffi import Curl, CurlInfo, CurlOpt
from typing_extensions import Optional

from syftbox.client.benchmark import Stats


@dataclass
class HTTPTimingStats:
    """Container for HTTP timing statistics."""

    dns: Stats
    """Time taken to resolve the host name"""
    tcp_connect: Stats
    """Time taken to establish a TCP connection"""
    ssl_handshake: Stats
    """Time taken to perform the SSL handshake"""
    send: Stats
    """Time taken to send the request"""
    server_wait: Stats
    """Time spent waiting for the server to send the first byte of the response"""
    content: Stats
    """Time taken to download the response"""
    total: Stats
    """Total time taken for the request"""
    redirect: Stats
    """Time taken for all redirection steps before the final request"""
    success_rate: float
    """Percentage of successful requests"""


@dataclass
class HTTPTimings:
    dns: float
    """Time taken to resolve the host name"""
    tcp_connect: float
    """Time taken to establish a TCP connection"""
    ssl_handshake: float
    """Time taken to perform the SSL handshake"""
    send: float
    """Time taken to send the request"""
    server_wait: float
    """Time spent waiting for the server to send the first byte of the response"""
    content: float
    """Time taken to download the response"""
    total: float
    """Total time taken for the request"""
    redirect: float
    """Time taken for all redirection steps before the final request"""


class HTTPPerfStats:
    """Measure HTTP connection performance using curl_cffi"""

    def __init__(self, url: str):
        self.url = url
        self.connect_timeout: int = 30
        self.total_timeout: int = 60
        self.max_redirects: int = 5

    def get_stats(self, n_runs: int) -> HTTPTimingStats:
        """Aggregate performance stats from multiple runs"""

        measurements: list[HTTPTimings] = []
        for _ in range(n_runs):
            if stats := self.__make_request(self.url):
                measurements.append(stats)
            time.sleep(0.5)  # Small delay between requests

        if not measurements:
            raise RuntimeError("No successful measurements")

        # Calculate aggregated stats
        def _stats_for_measurement(metric: str) -> Stats:
            values = [getattr(m, metric) for m in measurements]
            return Stats.from_values(values)

        return HTTPTimingStats(
            dns=_stats_for_measurement("dns"),
            tcp_connect=_stats_for_measurement("tcp_connect"),
            ssl_handshake=_stats_for_measurement("ssl_handshake"),
            send=_stats_for_measurement("send"),
            server_wait=_stats_for_measurement("server_wait"),
            content=_stats_for_measurement("content"),
            total=_stats_for_measurement("total"),
            redirect=_stats_for_measurement("redirect"),
            success_rate=len(measurements) / n_runs * 100,
        )

    def __make_request(self, url: str) -> Optional[HTTPTimings]:
        """Get HTTP performance stats for a single request"""

        buff = BytesIO()
        curl = Curl()

        opts = {
            CurlOpt.URL: url.encode(),
            CurlOpt.WRITEDATA: buff,
            CurlOpt.FOLLOWLOCATION: 1,
            CurlOpt.MAXREDIRS: self.max_redirects,
            CurlOpt.CONNECTTIMEOUT: self.connect_timeout,
            CurlOpt.TIMEOUT: self.total_timeout,
            CurlOpt.SSL_VERIFYPEER: 1,
            CurlOpt.SSL_VERIFYHOST: 2,
        }
        [curl.setopt(option, value) for option, value in opts.items()]

        try:
            curl.perform()

            # Curl Timings https://curl.se/libcurl/c/curl_easy_getinfo.html#TIMES
            # from start of request to stage (in microseconds)
            namelookup_t = curl.getinfo(CurlInfo.NAMELOOKUP_TIME)  # DNS lookup
            connect_t = curl.getinfo(CurlInfo.CONNECT_TIME)
            appconnect_t = curl.getinfo(CurlInfo.APPCONNECT_TIME)
            pretransfer_t = curl.getinfo(CurlInfo.PRETRANSFER_TIME)
            starttransfer_t = curl.getinfo(CurlInfo.STARTTRANSFER_TIME)  # TTFB
            total_t = curl.getinfo(CurlInfo.TOTAL_TIME)  # total time or TTLB
            redirect_t = curl.getinfo(CurlInfo.REDIRECT_TIME)

            # 1. Time spent resolving the host name
            dns = namelookup_t
            # 2. Time spent establishing a TCP connection
            tcp_connect = connect_t - namelookup_t
            # 3. Time spent performing the SSL handshake
            ssl_handshake = appconnect_t - connect_t
            # 4. Time spent sending the request
            send = pretransfer_t - appconnect_t
            # 5. Time spent waiting for server to send the first byte
            server_wait = starttransfer_t - pretransfer_t
            # 6. Time to download the response
            content = total_t - starttransfer_t

            return HTTPTimings(
                dns=dns * 1000,
                tcp_connect=tcp_connect * 1000,
                ssl_handshake=ssl_handshake * 1000,
                send=send * 1000,
                server_wait=server_wait * 1000,
                content=content * 1000,
                total=total_t * 1000,
                redirect=redirect_t * 1000,
            )
        except Exception as e:
            raise e
        finally:
            curl.close()
