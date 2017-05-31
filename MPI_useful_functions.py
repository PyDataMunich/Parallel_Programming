from mpi4py import MPI
import math, numpy, time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def custom_barrier(sel_ranks = None):
    """MPI barrier emulator with support of dead ranks"""
    if not sel_ranks:
        sel_ranks = list(range(size))
    if len(sel_ranks) == 1:
        return
    if rank == sel_ranks[0]:
        expected = len(sel_ranks) - 1
        got = 0
        while True:
            for rank_ in sel_ranks[1: ]:
                if comm.Iprobe(source = rank_, tag = 414):
                    first = None
                    first = comm.recv(first, source = rank_, tag = 414)
                    got += 1
            if got == expected:
                break
            time.sleep(0.1)
        for rank_ in sel_ranks[1: ]:
            comm.send("second", dest = rank_, tag = 414)
    else:
        comm.send("first", dest = sel_ranks[0], tag = 414)
        while True:
            if comm.Iprobe(source = sel_ranks[0], tag = 414):
                second = None
                second = comm.recv(second, source = sel_ranks[0], tag = 414)
                break
            time.sleep(0.1)

def bcast_array(array = None, sel_ranks = None, root = 0):
    """Function for broadcasting a NumPy array to selected ranks"""
    if rank != root:
        expected_len = None
        expected_len = comm.bcast(expected_len, root = root)
        sel_ranks = None
        sel_ranks = comm.bcast(sel_ranks, root = root)
        array = None
        while True:
            err = False
            try:
                part_array = None
                part_array = comm.recv(part_array, source = root, tag = 1)
            except (MPI.Exception, SystemError, EOFError):
                err = True
                array = None
            if rank == min(sel_ranks):
                comm.send(err, dest = root, tag = 2)
            if err:
                continue
            if str(type(part_array)) == '<class \'numpy.ndarray\'>':
                if type(array) == type(None):
                    array = numpy.zeros(0, dtype = part_array.dtype)
                len_ = len(array)
                array.resize(len_ + len(part_array))
                array[len_: ] = part_array
                len_ = len(array)
            else:
                break
        if len_ != expected_len:
            raise ValueError('Error while array receiving: got an array of \
length {} instead of {}'.format(len_, expected_len))
            ##
        return array
    elif rank == root:
        array_len = len(array)
        if not sel_ranks:
            raise Valueerror(\
            'List of selected ranks must be provided in the root rank')
            ##
        if str(type(array)) != '<class \'numpy.ndarray\'>':
            raise TypeError('A NumPy array must be provided in source rank')
        sel_ranks = set(sel_ranks)
        if rank in sel_ranks:
            sel_ranks.remove(rank)
        if len(sel_ranks) == 0:
            return
        comm.bcast(array_len, root = 0)
        comm.bcast(sel_ranks, root = 0)
        array_size = array.dtype.itemsize * array_len
        parts = max(math.ceil(array_size * 1.1 / 2 ** 31), 1)
        err = False
        part_len = math.ceil(array_len / parts)
        for part in range(parts):
            part_begin = part * part_len
            part_end = min((part + 1) * part_len, array_len)
            try:
                for i in sel_ranks:
                    comm.send(array[part_begin: part_end], dest = i, tag = 1)
            except MPI.Exception:
                err = True
            if not err:
                err = None
                err = comm.recv(err, source = min(sel_ranks), tag = 2)
            if err:
                raise RuntimeError('Failed to broadcast the array')
        for i in sel_ranks:
            comm.send(None, dest = i, tag = 1)
        err = None
        err = comm.recv(err, source = min(sel_ranks), tag = 2)

def sendrecv_array(array = None, source = 0, dest = 0):
    """Function for sending a NumPy array between two processes"""
    if source == dest:
        raise ValueError('Source and destination cannot be the same')
    if rank == dest:
        expected_len = None
        expected_len = comm.recv(expected_len, source = source)
        array = None
        while True:
            err = False
            try:
                part_array = None
                part_array = comm.recv(part_array, source = source, tag = 1)
            except (MPI.Exception, SystemError, EOFError):
                err = True
                array = None
            comm.send(err, dest = source, tag = 2)
            if err:
                continue
            if str(type(part_array)) == '<class \'numpy.ndarray\'>':
                if type(array) == type(None):
                    array = numpy.zeros(0, dtype = part_array.dtype)
                len_ = len(array)
                array.resize(len_ + len(part_array))
                array[len_: ] = part_array
                len_ = len(array)
            else:
                break
        if len_ != expected_len:
            raise ValueError('Error while array receiving: got an array of \
length {} instead of {}'.format(len_, expected_len))
            ##
        return array
    if rank == source:
        array_len = len(array)
        comm.send(array_len, dest = dest)
        if str(type(array)) != '<class \'numpy.ndarray\'>':
            raise TypeError('A NumPy array must be provided in source rank')
        array_size = array.dtype.itemsize * array_len
        parts = max(math.ceil(array_size * 1.1 / 2 ** 31), 1)
        err = False
        part_len = math.ceil(array_len / parts)
        for part in range(parts):
            part_begin = part * part_len
            part_end = min((part + 1) * part_len, array_len)
            try:
                comm.send(array[part_begin: part_end],
                          dest = dest, tag = 1)
            except MPI.Exception:
                err = True
            if not err:
                err = comm.recv(source = dest, tag = 2)
            if err:
                break
        if err:
            raise RuntimeError('Failed to send the array')
        comm.send(None, dest = dest, tag = 1)
        err = comm.recv(source = dest, tag = 2)