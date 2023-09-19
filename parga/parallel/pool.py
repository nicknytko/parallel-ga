import sys
import inspect

import parga.parallel.single_process
import parga.parallel.multiprocess
import parga.parallel.mpi

from parga.parallel.worker import WorkerCommand, worker_process


class WorkerPool:
    '''
    A pool of workers that can either be spun up as child processes or inferred from MPI tasks
    '''

    class NotMainProcess(ValueError):
        pass


    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.workers = []
        self.mpi_enabled = False
        self.started = False
        self.is_main = True


    def __getitem__(self, key):
        return self.workers[key]


    def __len__(self):
        return self.num_workers


    def start(self):
        '''
        Creates and starts workers for the pool.
        '''
        if self.started:
            return

        # The thread of execution looks slightly different if we're using multiprocessing vs MPI
        # MPI:
        #  All processes will enter this function.  If we're the main (rank=0) process, we create
        #  worker objects for every other process, then proceed normally.  If we're not the main
        #  process, we enter the worker loop and directly exit afterwards.
        # Multiprocessing:
        #  The main process will enter this function.  It then will create however many child
        #  processes that were specified; the child processes will enter this function and then
        #  immediately exit.  Then, it will reach the end of the main script and enter the main loop.

        # Check for MPI
        parga.parallel.mpi.mpi_initialize()
        if parga.parallel.mpi.mpi_enabled():
            self.num_workers = parga.parallel.mpi.mpi_get_num_workers()
            self.mpi_enabled = True
            rank = parga.parallel.mpi.mpi_rank()

            # Enter the worker loop if we aren't the main process
            if rank != 0:
                self.is_main = False
                try:
                    worker_process(parga.parallel.mpi.MPIPipeEnd(None, rank, 0))
                finally:
                    parga.parallel.mpi.mpi_finalize()
                    sys.exit(0)
        elif parga.parallel.multiprocess.mp_is_subprocess():
            self.is_main = False
            return

        # Create all workers
        for i in range(self.num_workers):
            if i == 0:
                self.workers.append(parga.parallel.single_process.SingleProcessWorker())
            else:
                if self.mpi_enabled:
                    self.workers.append(parga.parallel.mpi.MPIWorker(i))
                else:
                    self.workers.append(parga.parallel.multiprocess.MultiprocessWorker())

        # Start all workers
        for worker in self.workers:
            worker.start()

        self.started = True


    def receive_all(self):
        '''
        Gathers data from all worker processes.

        Returns
        -------
        data - python list
          List of data, such that data[i] is what was received from worker i.
        '''

        data = []
        for worker in self.workers:
            data.append(worker.receive())
        return data


    def finish(self):
        '''
        Closes all workers in the queue, and frees associated resources.
        '''

        if not self.started:
            raise RuntimeError('Cannot close WorkerQueue: not started')

        for worker in self.workers:
            worker.finish()

        parga.parallel.mpi.mpi_finalize()


    def __enter__(self):
        self.start()
        if not self.mpi_enabled and parga.parallel.multiprocess.mp_is_subprocess():
            # if we're a child multiprocess, then exit and skip execution of the 'with' block
            def raise_not_main_process(*args):
                raise WorkerPool.NotMainProcess()

            sys.settrace(lambda *args, **keys: None)
            frame = inspect.currentframe().f_back
            frame.f_trace = raise_not_main_process
        return self


    def __exit__(self, type, value, tb):
        # Return true if no error or we get the "not main process" error
        # (which is not actually an error, but a hack to skip execution of the body)
        if self.is_main:
            self.finish()
        return (type is None or type == WorkerPool.NotMainProcess)


    def map(self, iterable, function, extra_args=None):
        '''
        Maps an iterable object onto a function, returning a list.

        Parameters
        ----------
        iterable : iterable
          An object that can be iterated through, for example a list, numpy array,
          or a tuple.
        function : callable
          A function of the form:

          f(x_i, *extra_args)

          where x_i is the current iterable element and *extra_args are any arguments
          specified in extra_args.
        extra_args : tuple
          Any extra arguments that should be passed to 'function'

        Returns
        -------
        mapped_iterable : list
          A list where every element is the return of 'function', i.e. of the form
          [ f(iterable[0], *extra_args), f(iterable[1], *extra_args), ... ]
        '''
        if isinstance(iterable, int):
            iterable = range(iterable)
        output = [None]*len(iterable)

        for i, worker in enumerate(self.workers):
            local_iterable = iterable[i::self.num_workers]
            if len(local_iterable) != 0:
                worker.send_command(WorkerCommand.create(WorkerCommand.MAP,
                                                         iterable=local_iterable,
                                                         function=function,
                                                         worker_idx=i,
                                                         args=extra_args))
            else:
                worker.send_command(WorkerCommand.create(WorkerCommand.NOOP))

        # Now, assemble data we get back from the workers
        data = self.receive_all()
        for datum in data:
            if datum['command'] == WorkerCommand.NOOP:
                continue
            output[datum['worker_idx']::self.num_workers] = datum['output']

        return output
