import argparse
import tensorflow as tf

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=int, default=0, help='the task number')
args = parser.parse_args()

cluster = tf.train.ClusterSpec({'local': ['localhost:2222', 'localhost:2223']})
server = tf.train.Server(cluster, job_name='local', task_index=args.task)
server.start()
server.join()
