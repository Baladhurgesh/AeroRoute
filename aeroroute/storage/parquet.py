from collections import Counter, OrderedDict
from pathlib import Path

import pyarrow.parquet as pq

from .artifacts import contained_path


class RowGroupCache:
    def __init__(self, root, files, cache_bytes=64 * 1024 ** 2, max_handles=16):
        if type(cache_bytes) is not int or cache_bytes < 0 or type(max_handles) is not int or max_handles < 1:
            raise ValueError("Invalid cache limits")
        self.root, self.files = Path(root), files
        self.cache_bytes, self.max_handles = cache_bytes, max_handles
        self.groups, self.handles = OrderedDict(), OrderedDict()
        self.cache_size_bytes = 0
        self.metrics = Counter()

    def _handle(self, file_id):
        item = self.files[file_id]
        handle = self.handles.pop(file_id, None)
        if handle is None:
            if len(self.handles) >= self.max_handles:
                self.handles.popitem(last=False)[1].close()
            handle = pq.ParquetFile(contained_path(self.root, item["path"]))
        self.handles[file_id] = handle
        return handle

    def _stream_row(self, handle, group, offset, columns):
        seen = 0
        for batch in handle.iter_batches(batch_size=8192, row_groups=[group], columns=columns):
            self.metrics["decoded_bytes"] += batch.nbytes
            self.metrics["decoded_rows"] += batch.num_rows
            if seen + batch.num_rows > offset:
                return batch.slice(offset - seen, 1).to_pylist()[0]
            seen += batch.num_rows
        raise ValueError("Row offset out of bounds")

    def row(self, file_id, group, offset, columns=None):
        item = self.files[file_id]
        key = (item["sha256"], group, tuple(columns) if columns is not None else None)
        if key in self.groups:
            table = self.groups.pop(key)
            self.groups[key] = table
            self.metrics["cache_hits"] += 1
            if not 0 <= offset < table.num_rows:
                raise ValueError("Row offset out of bounds")
            return table.slice(offset, 1).to_pylist()[0]
        self.metrics["cache_misses"] += 1
        handle = self._handle(file_id)
        if not 0 <= group < handle.num_row_groups:
            raise ValueError("Row group locator out of bounds")
        stored = handle.metadata.row_group(group).total_byte_size
        if stored > self.cache_bytes:
            self.metrics["row_groups_read"] += 1
            self.metrics["row_groups_streamed"] += 1
            return self._stream_row(handle, group, offset, columns)
        table = handle.read_row_group(group, columns=columns)
        self.metrics["row_groups_read"] += 1
        self.metrics["decoded_bytes"] += table.nbytes
        self.metrics["decoded_rows"] += table.num_rows
        if table.nbytes <= self.cache_bytes:
            while self.groups and self.cache_size_bytes + table.nbytes > self.cache_bytes:
                self.cache_size_bytes -= self.groups.popitem(last=False)[1].nbytes
            self.groups[key] = table
            self.cache_size_bytes += table.nbytes
        if not 0 <= offset < table.num_rows:
            raise ValueError("Row offset out of bounds")
        return table.slice(offset, 1).to_pylist()[0]

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()
        self.groups.clear()
        self.cache_size_bytes = 0
