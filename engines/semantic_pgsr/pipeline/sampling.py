import random


class CameraSampler:
    def __init__(self, cameras, instance_ids, seed=22):
        self.cameras = cameras
        self.rng = random.Random(seed)
        self.eligible = {i: [j for j,c in enumerate(cameras) if (i in c.available_instance_ids if getattr(c,'available_instance_ids',None) is not None else c.instance_mask is None or (c.instance_mask == i).any())]
                         for i in instance_ids}
        if any(not rows for rows in self.eligible.values()):
            raise ValueError('Every selected instance must have at least one eligible camera')
        self.queues = {i: [] for i in instance_ids}
        self.instances = list(instance_ids)
        self.order = []

    def sample(self, count):
        if not self.order:
            self.order = self.instances.copy()
            self.rng.shuffle(self.order)
        instance = self.order.pop()
        queue = self.queues[instance]
        if not queue:
            queue.extend(self.eligible[instance])
            self.rng.shuffle(queue)

        return instance, [queue.pop() for _ in range(min(count,len(queue)))]

    def sample_pairs(self, count):

        if count < 1:
            raise ValueError('count must be positive')
        pairs = []
        for _ in range(count):
            if not self.order:
                self.order = self.instances.copy()
                self.rng.shuffle(self.order)
            instance = self.order.pop()
            queue = self.queues[instance]
            if not queue:
                queue.extend(self.eligible[instance])
                self.rng.shuffle(queue)
            pairs.append((instance, queue.pop()))
        return pairs

    def capture(self):
        return {'random': self.rng.getstate(), 'queues': self.queues, 'order': self.order}

    def restore(self, state):
        self.rng.setstate(state['random'])
        self.queues = state['queues']
        self.order = state['order']
