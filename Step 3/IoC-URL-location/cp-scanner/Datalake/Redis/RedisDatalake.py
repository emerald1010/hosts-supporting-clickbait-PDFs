import pickle

from Datalake.DataLake import DataLake
from redis import Redis


class RedisDatalake(DataLake):

    def __init__(self, host, port, database):
        self.redis_conn = Redis(host, port, database, retry_on_timeout=True, socket_timeout=10)

    @staticmethod
    def prepare_datalake(config):
        specific_conf = config["global"]['processes']['redis']
        return RedisDatalake(specific_conf['host'], specific_conf['port'], specific_conf['database'])

    def create_queue(self, queue_name):
        """
        Creates a Redis queue object
        :param queue_name: name of the queue
        :return:
        """
        return RedisQueue(self.redis_conn, queue_name)

    def __del__(self):
        if self.redis_conn:
            self.redis_conn.close()

class RedisQueue:

    def __init__(self, conn: Redis, key):
        self.conn = conn
        self.key = key

    def push(self, values):
        self.conn.lpush(self.key, pickle.dumps(values))

    def pop(self):
        e = self.conn.rpop(self.key)

        # if no element is in the queue return false
        if not e:
            return False

        e = pickle.loads(e)
        return e

    def ping(self):
        return self.conn.ping()

    def clean(self):
        return self.conn.delete(self.key)

    def count(self):
        return self.conn.llen(self.key)

