import argparse

parser = argparse.ArgumentParser(description='Poll Openmined for Jobs.')
parser.add_argument('--interval', dest='interval', action='store',
                   default=10, type=int,
                   help='How often to poll for jobs in seconds (default: 10)')

args = parser.parse_args()

from grid.server.grid import Server as GridWorker

w = GridWorker()
w.poll(args.interval)
