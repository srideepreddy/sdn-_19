import eventlet
# This is the "magic" fix for Ryu + Eventlet in newer Python
# We must use thread=False to avoid the TypeError in is_timeout
eventlet.monkey_patch(all=True, thread=False)

import os
os.environ['EVENTLET_NO_GREENDNS'] = 'yes'

# Aggressive patch for Ryu's wsgi
try:
    import ryu.app.wsgi as ryu_wsgi
    import eventlet.wsgi
    # Some versions of ryu try to monkey patch again.
    # We can try to prevent that by mocking the monkey_patch function
    eventlet.monkey_patch = lambda *args, **kwargs: None
except ImportError:
    pass

from ryu.cmd.manager import main
import sys

if __name__ == "__main__":
    sys.exit(main())
