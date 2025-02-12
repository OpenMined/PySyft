import json
import time
import base64
import requests
import cbor2

from syft_rpc import SyftBoxURL


class SyftBoxRPCSession(requests.Session):
    def __init__(self, additional_headers=None, syftbox_proxy=None):
        super().__init__()
        self.additional_headers = additional_headers or {}
        self.syftbox_proxy = syftbox_proxy

    def request(self, method, url, headers=None, *args, **kwargs):
        print("kwargs", kwargs)
        if self.syftbox_proxy is None:
            raise Exception(f"{type(self)} requires a syftbox_proxy")
        # Merge additional headers with existing headers
        headers = {**self.additional_headers, **(headers or {})}

        if "syft://" in str(url):
            syft_url = SyftBoxURL(url)
            params = {"method": method} | syft_url.as_http_params()
            if "timeout" in kwargs:
                params["timeout"] = kwargs["timeout"]
        else:
            raise Exception(f"{type(self)} requires a syft:// url")

        print("calling", self.syftbox_proxy, params)
        response = super().request(
            "post", self.syftbox_proxy, params=params, headers=headers, *args, **kwargs
        )
        response.decoded_content = self.decode(response)
        response.wait = self._create_wait_method(response)
        response.retry = self._create_retry_method(
            response, method, url, headers, args, kwargs, self
        )
        return response

    @staticmethod
    def decode(response):
        """Decode the response based on its Content-Type."""
        content_type = response.headers.get("content-type", "")
        try:
            if content_type == "application/json":
                # JSON response, possibly base64 encoded
                try:
                    decoded_content = base64.b64decode(response.content)
                    return json.loads(decoded_content)
                except Exception:
                    # If decoding fails, fallback to direct JSON parsing
                    return response.json()
            elif content_type == "application/cbor":
                # CBOR response
                return cbor2.loads(response.content)
            else:
                # Default to UTF-8 decoding
                return response.content.decode("utf-8")
        except Exception as e:
            raise Exception(f"Failed to decode response: {e}") from e

    def _create_wait_method(self, response):
        """Attach a wait method to the response object."""

        def wait(
            location_header="Location",
            max_retries=10,
            time_between_retries=5,
            timeout=60,
        ):
            if response.status_code == 200:
                return response
            if response.status_code != 202:
                raise Exception("You need a 202 status code to wait on.")

            location = response.headers.get(location_header)
            if not location:
                raise ValueError(f"Response missing '{location_header}' header")

            # Extract the base URL (host and scheme) from the original response URL
            base_url = "/".join(response.url.split("/")[:3])
            full_location = (
                location if location.startswith("http") else f"{base_url}{location}"
            )

            print(f"202 received. Polling location: {full_location}")
            start_time = time.time()
            retries = 0

            while retries < max_retries and (time.time() - start_time) < timeout:
                # Use self.request to leverage custom logic
                poll_response = requests.get(full_location)
                if poll_response.status_code == 200:
                    print("200 OK received. Request successful.")

                    # Mutate the original response object
                    response.status_code = poll_response.status_code
                    response.headers = poll_response.headers
                    response._content = poll_response.content
                    response.decoded_content = self.decode(poll_response)

                    return response  # Return the mutated response object

                print(
                    f"Polling attempt {retries + 1}: Received {poll_response.status_code}. Retrying in {time_between_retries}s..."
                )
                time.sleep(time_between_retries)
                retries += 1

            raise TimeoutError("Timed out waiting for a 200 response.")

        return wait

    def _create_retry_method(
        self, response, method, url, headers, args, kwargs, session
    ):
        """Attach a retry method to the response object."""

        def retry():
            if response.status_code == 200:
                return response
            if response.status_code != 504:
                raise Exception(
                    "Retry is only allowed for responses with a 504 status code."
                )

            print("Retrying the original request...")
            # Resend the same request using the session's custom request method
            retried_response = session.request(
                method, url, headers=headers, *args, **kwargs
            )

            # Mutate the original response object with the new response's attributes
            response.status_code = retried_response.status_code
            response.headers = retried_response.headers
            response._content = retried_response.content
            response.decoded_content = retried_response.decoded_content

            # Update wait and retry methods on the mutated response
            response.wait = self._create_wait_method(response)
            response.retry = self._create_retry_method(
                response, method, url, headers, args, kwargs, session
            )

            return response  # Return the mutated original response

        return retry
