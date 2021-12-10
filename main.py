import numpy as np
from mpmath import nsum, nprod, fac
from statistics import mean
import matplotlib.pyplot as plt


class QueryArrived:
    def __init__(self, timepoint):
        self.timepoint = timepoint

    def __str__(self):
        return self.timepoint


class QueryProcessed:
    def __init__(self, start_timepoint, timepoint, channel):
        self.start_timepoint = start_timepoint
        self.timepoint = timepoint
        self.channel = channel

    def __str__(self):
        return self.timepoint


class Model:
    def __init__(self, n, lambd, apt, awt, queries, eps):
        self.n = n
        self.m = 10
        self.lambd = lambd
        self.mu = 1/apt
        self.nu = 1/awt
        self.queries = queries
        self.eps = eps

        self.channel_availabilities = [True] * n
        self.timeline = []
        self.queue = []

        self.busy_channels = 0

        self.state_durations = []
        self.final_state_durations = [0] * (n + self.m + 1)
        self.last_state = 0
        self.last_state_change_timepoint = 0

        self.queries_processed = 0
        self.queries_dropped = 0

        self.processing_time = []
        self.waiting_time = []

    def start(self):
        query_gen = self.generate(self.lambd)
        processing_gen = self.generate(self.mu)
        waiting_gen = self.generate(self.nu)

        first_timepoint = next(query_gen)
        self.timeline.append(QueryArrived(first_timepoint))

        for _ in range(self.queries - 1):
            timepoint = self.timeline[len(self.timeline) - 1].timepoint + next(query_gen)
            self.timeline.append(QueryArrived(timepoint))

        for event in self.timeline:
            self.clean_queue(event.timepoint)

            if isinstance(event, QueryArrived):
                has_available_channels, channel = self.find_channel()

                if has_available_channels:
                    query_processed_timepoint = event.timepoint + next(processing_gen)
                    query_processed_event = QueryProcessed(event.timepoint, query_processed_timepoint, channel)
                    self.insert(query_processed_event)
                    self.record_state(event.timepoint)
                    self.waiting_time.append(0)
                else:
                    waiting_deadline = event.timepoint + next(waiting_gen)
                    self.queue.append((event.timepoint, waiting_deadline))
                    self.record_state(event.timepoint)

            if isinstance(event, QueryProcessed):
                self.queries_processed += 1
                self.busy_channels -= 1
                self.channel_availabilities[event.channel] = True
                self.processing_time.append(event.timepoint - event.start_timepoint)

                if len(self.queue) != 0:
                    start_timepoint, _ = self.queue.pop(0)

                    query_processed_timepoint = event.timepoint + next(processing_gen)
                    query_processed_event = QueryProcessed(event.timepoint, query_processed_timepoint,
                                                           event.channel)
                    self.channel_availabilities[event.channel] = False
                    self.busy_channels += 1
                    self.insert(query_processed_event)
                    self.waiting_time.append(event.timepoint - start_timepoint)

                self.record_state(event.timepoint)

    def generate(self, param):
        while True:
            yield np.random.exponential(1 / param)

    def insert(self, event):
        for i in range(1, len(self.timeline)):
            if self.timeline[i - 1].timepoint < event.timepoint < self.timeline[i].timepoint:
                self.timeline.insert(i, event)
                break
        else:
            self.timeline.append(event)

    def find_channel(self):
        for i, status in enumerate(self.channel_availabilities):
            if status:
                self.busy_channels += 1
                self.channel_availabilities[i] = False

                return True, i

        return False, -1

    def clean_queue(self, current_timepoint):
        cleaned_queue = []

        while len(self.queue) != 0:
            start_timepoint, waiting_deadline = self.queue.pop(0)

            if current_timepoint > waiting_deadline:
                self.queries_dropped += 1
                self.record_state(waiting_deadline)

            else:
                cleaned_queue.append((start_timepoint, waiting_deadline))

        self.queue = cleaned_queue

    def record_state(self, timepoint):
        time_delta = timepoint - self.last_state_change_timepoint
        self.final_state_durations[self.last_state] += time_delta

        self.last_state_change_timepoint = timepoint
        self.last_state = self.busy_channels + len(self.queue)

        self.state_durations.append((timepoint, self.final_state_durations.copy()))

    def show_empirical_stats(self):
        print('EMPIRICAL STATS')
        print('Queries processed:', self.queries_processed)
        print('Queries dropped:', self.queries_dropped)

        avg_processing = 0
        probs = []
        for k in range(self.n + 1):
            pk = self.final_state_durations[k] / self.timeline[-1].timepoint
            probs.append(pk)
            avg_processing += k * pk
            print(f'{k} channels are busy & 0 queries in queue | p{k} =', pk)

        avg_queue_length = 0
        s = self.n + 1
        while True:
            ps = self.final_state_durations[s] / self.timeline[-1].timepoint
            if ps < 0.001:
                break
            probs.append(ps)
            avg_queue_length += (s - self.n) * ps
            print(f'{self.n} channels are busy & {s - self.n} queries in queue | pn+{s - self.n} =', ps)
            s += 1

        p_denial = self.queries_dropped / self.queries
        Q = 1 - p_denial
        A = self.lambd * Q
        avg_waiting_time = mean(self.waiting_time)
        avg_processing_time = mean(self.processing_time)
        avg_total_time = avg_waiting_time + avg_processing_time
        print('Probability of denial: ', p_denial)
        print('Relative throughput: ', Q)
        print('Absolute throughput', A)
        print('Average queue length: ', avg_queue_length)
        print('Average processing queries: ', avg_processing)
        print('Average queries in system: ', avg_processing + avg_queue_length)
        print('Average processing time: ', avg_processing_time)
        print('Average waiting time: ', avg_waiting_time)
        print('Average time in system: ', avg_total_time)

        return probs, p_denial, Q, A, avg_processing, avg_queue_length, avg_processing + avg_queue_length, \
               avg_processing_time, avg_waiting_time, avg_total_time

    def show_theoretical_stats(self):
        print('THEORETICAL STATS')
        ro = self.lambd / self.mu
        p0 = 1 / (nsum(lambda k: ro ** k / fac(k), [0, self.n]) + (ro ** self.n / fac(self.n)) * nsum(
            lambda i: self.lambd ** i / nprod(lambda l: (self.n * self.mu + l * self.nu), [1, i]), [1, self.m]))
        print('0 channels are busy & 0 queries in queue | p0 =', p0)

        avg_processing = 0
        probs = [p0]
        for k in range(1, self.n + 1):
            pk = ro ** k * p0 / fac(k)
            probs.append(pk)
            avg_processing += k * pk
            print(f'{k} channels are busy & 0 queries in queue | p{k} =', pk)

        avg_queue_length = 0
        s = 1
        while True:
            ps = (ro ** self.n) * (self.lambd ** s) * p0 / (fac(self.n) * nprod(lambda l: self.n * self.mu + l * self.nu, [1, s]))
            if ps < 0.001:
                break
            probs.append(ps)
            avg_queue_length += s * ps
            print(f'{self.n} channels are busy & {s} queries in queue | pn+{s} =', ps)
            s += 1

        p_denial = 1 - (self.mu / self.lambd) * (avg_processing + avg_queue_length)
        Q = 1 - p_denial
        A = self.lambd * Q
        avg_waiting_time = avg_queue_length / self.lambd
        avg_processing_time = 1 / self.mu
        avg_total_time = avg_processing_time + avg_waiting_time
        print('Probability of denial: ', p_denial)
        print('Relative throughput: ', Q)
        print('Absolute throughput', A)
        print('Average queue length: ', avg_queue_length)
        print('Average processing queries: ', avg_processing)
        print('Average queries in system: ', avg_processing + avg_queue_length)
        print('Average processing time: ', avg_processing_time)
        print('Average waiting time: ', avg_waiting_time)
        print('Average time in system: ', avg_total_time)

        return probs, p_denial, Q, A, avg_processing, avg_queue_length, avg_processing + avg_queue_length, \
               avg_processing_time, avg_waiting_time, avg_total_time

    def show_stationary_stats(self):
        x = [timepoint for timepoint, _ in self.state_durations]

        for i in range(len(self.final_state_durations)):
            y = [durations[i] / timepoint for timepoint, durations in self.state_durations]
            plt.plot(x, y)

        plt.show()

    def show_stats(self):
        print('______________________________________________')
        e_probs, e_p_denial, e_Q, e_A, e_avg_processing, e_avg_queue_length, e_avg_total, e_avg_processing_time, \
        e_avg_waiting_time, e_avg_total_time = model.show_empirical_stats()
        print()
        t_probs, t_p_denial, t_Q, t_A, t_avg_processing, t_avg_queue_length, t_avg_total, t_avg_processing_time, \
        t_avg_waiting_time, t_avg_total_time = model.show_theoretical_stats()

        assert abs(e_p_denial - t_p_denial)
        assert abs(e_Q - t_Q) < self.eps
        assert abs(e_avg_processing - t_avg_processing) < self.eps
        assert abs(e_avg_queue_length - t_avg_queue_length) < self.eps
        assert abs(e_avg_total - t_avg_total) < self.eps
        assert abs(e_avg_processing_time - t_avg_processing_time) < self.eps
        assert abs(e_avg_waiting_time - t_avg_waiting_time) < self.eps
        assert abs(e_avg_total_time - t_avg_total_time) < self.eps
        print('MODEL IS CORRECT')

        _, ax = plt.subplots(1, 2)
        ax[0].title.set_text(
            f'Empirical probabilities')
        ax[0].bar(list(np.arange(0, len(e_probs))), e_probs)
        ax[1].title.set_text(
            f'Theoretical probabilities')
        ax[1].bar(list(np.arange(0, len(t_probs))), t_probs)
        plt.show()
        print(e_probs)
        model.show_stationary_stats()

        print('______________________________________________')


if __name__ == '__main__':

    model = Model(2, 3, 1, 0.5, 1000, 0.1)
    model.start()
    model.show_stats()
