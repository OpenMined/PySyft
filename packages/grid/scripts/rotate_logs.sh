#!/bin/bash

# cronjob logs: $ tail -f /var/log/syslog | grep -i cron

journalctl --rotate
journalctl --vacuum-size=100M
journalctl --disk-usage
