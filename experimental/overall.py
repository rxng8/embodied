# %%

import embodied

from embodied.nn import SimpleEncoder

a = SimpleEncoder({}, act="hello", name="a")
b = SimpleEncoder({}, act="b", name="b")
SimpleEncoder.act

# %%

from embodied.api import make_logger
make_logger(embodied.Config(logdir="foo", filter=".*"))
