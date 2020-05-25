# Some functions to aid monitoring network traffic

import pyshark


class NetworkMonitor:
    """
    Provides utility for the monitoring of network traffic, measuring the packets sent through
    specific filters as passed by user.
    """

    @staticmethod
    def get_packets(
        timeout=50,
        interface=None,
        bpf_filter=None,
        display_filter="tcp.port == 80",
        tshark_path=None,
        output_file=None,
    ):
        """
        Returns the captured packets of the transmitted data using Wireshark.

        Args:
        timeout: An integer. Set for sniffing with tshark. Default to 50 seconds in this setup.
        interface: A string. Name of the interface to sniff on.
        bpf_filter: A string. The capture filter in bpf syntax 'tcp port 80'. Needs to be changed
                    to match filter for the traffic sent. Not to be confused with the display
                    filters (e.g. tcp.port == 80). The former are much more limited and is used to
                    restrict the size of a raw packet capture, whereas the latter is used to hide
                    some packets from the packet list. More info can be found at
                    https://wiki.wireshark.org/CaptureFilters.
        display_filter: A string. Default to 'tcp.port == 80' (assuming this is the port of the
                        'WebsocketClientWorker'). Please see notes for 'bpf_filter' for details
                        regarding differences. More info can be found at
                        https://wiki.wireshark.org/DisplayFilters.
        tshark_path: Path to the tshark binary. E.g. '/usr/local/bin/tshark'.
        output_file: A string. Path including the output file name is to saved.
                     E.g. '/tmp/mycapture.cap'

        Returns:
        catpure: A 'pyshark.capture.live_capture.LiveCapture' object. Of packets sent
                 over WebSockets.
        length: An integer. The number of packets captured at the network interface.
        """
        capture_output = []
        if interface is None:
            raise Exception("Please provide the interface used.")
        else:
            capture = pyshark.LiveCapture(
                interface=interface,
                bpf_filter=bpf_filter,
                tshark_path=tshark_path,
                output_file=output_file,
            )
            capture.sniff(timeout=timeout)
            length = len(capture)
            return capture, length

    @staticmethod
    def read_packet(index=None, capture_input=None):
        """
        Reads the info of one packet returned by get_packets using pretty_print().

        Args:
        index: An integer. The index of the packet to be examined.

        Returns:
        pretty_print: The info of the packet chosen to be read.
        """
        if index is None:
            raise Exception(
                "Please choose an index within the total number of packets captured by get_packets."
            )
        elif capture_input is None:
            raise Exception("Please input the capture_output from get_packets.")
        elif not isinstance(index, int):
            raise Exception("The index passed is not an integer.")
        else:
            length = len(capture_input)
            if index < length:
                try:
                    packet = capture_input[index]
                    return packet.pretty_print()
                except:
                    raise Exception("Something went wrong when retrieving packet data.")
            else:
                raise Exception("The index given is not valid.")
