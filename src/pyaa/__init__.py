"""
┌───────────────────────────────────────────┐
│ Simple wrapper for aardvark C lib binding │
└───────────────────────────────────────────┘

 Florian Dupeyron
 April 2022
"""

import logging
import contextlib

from array       import array
from dataclasses import dataclass
from threading   import Event, RLock

from aardvark_py import (
    AA_PORT_NOT_FREE,
    AA_UNABLE_TO_OPEN,
    AA_INCOMPATIBLE_DEVICE,

    AA_FEATURE_SPI,
    AA_FEATURE_I2C,
    AA_FEATURE_GPIO,

    AA_CONFIG_GPIO_ONLY,
    AA_CONFIG_SPI_GPIO,
    AA_CONFIG_GPIO_I2C,
    AA_CONFIG_SPI_I2C,
    AA_CONFIG_QUERY,

    AA_I2C_PULLUP_NONE,
    AA_I2C_PULLUP_BOTH,
    AA_I2C_PULLUP_QUERY,

    AA_TARGET_POWER_NONE,
    AA_TARGET_POWER_BOTH,
    AA_TARGET_POWER_QUERY,

    AA_ASYNC_NO_DATA,
    AA_ASYNC_I2C_READ,
    AA_ASYNC_I2C_WRITE,
    AA_ASYNC_SPI,

    aa_find_devices_ext,
    aa_status_string,
    aa_open_ext,
    aa_close,
    aa_configure,
    aa_features,
    aa_target_power,
    aa_async_poll,

    aa_i2c_read_ext,
    aa_i2c_read,
    aa_i2c_write,
    aa_i2c_write_ext,
    
    aa_i2c_slave_enable,
    aa_i2c_slave_disable,
    aa_i2c_slave_read,
    aa_i2c_slave_read_ext,
    aa_i2c_bitrate,
    aa_i2c_pullup,
    aa_i2c_bus_timeout
)

# ┌────────────────────────────────────────┐
# │ Various constants                      │
# └────────────────────────────────────────┘

MAX_I2C_READ_SIZE=65535 # From aardvark manual


# ┌────────────────────────────────────────┐
# │ Error class                            │
# └────────────────────────────────────────┘

class PyAA_Probe_Error(Exception):
    def __init__(self, code, msg):
        super().__init__(f"{msg}: {aa_status_string(code)}")
        self.code = code


# ┌────────────────────────────────────────┐
# │ Probes list                            │
# └────────────────────────────────────────┘

@dataclass
class PyAA_ProbesList_Info:
    dev_id: int
    port_number: int
    free: bool

def list_probes(max_number_of_devices: int = 8):
    status, devices, ids = aa_find_devices_ext(max_number_of_devices, max_number_of_devices)

    return [
        PyAA_ProbesList_Info(
            dev_id      = dev_id,
            port_number = int     (dev_port & ~(AA_PORT_NOT_FREE)),
            free        = not(bool(dev_port &  (AA_PORT_NOT_FREE)))
        )
        for dev_port, dev_id in zip(devices, ids)
    ]

# ┌────────────────────────────────────────┐
# │ Probe managment                        │
# └────────────────────────────────────────┘

@dataclass
class PyAA_Features:
    i2c: bool
    spi: bool
    gpio: bool

    @classmethod
    def from_byte(cls, v):
        return cls(
            i2c  = bool(v & AA_FEATURE_I2C ),
            spi  = bool(v & AA_FEATURE_SPI ),
            gpio = bool(v & AA_FEATURE_GPIO)
        )

    def byte(self):
        return int(
            int(self.i2c )*AA_FEATURE_I2C,
            int(self.spi )*AA_FEATURE_SPI,
            int(self.gpio)*AA_FEATURE_GPIO
        )

@dataclass
class PyAA_Config:
    spi_enabled: bool
    i2c_enabled: bool

    @classmethod
    def from_byte(cls, v):
        return cls(
            spi_enabled = bool(v & AA_CONFIG_SPI_GPIO),
            i2c_enabled = bool(v & AA_CONFIG_GPIO_I2C)
        )

    def byte(self):
        return int(
            int(self.spi_enabled) * AA_CONFIG_SPI_GPIO |
            int(self.i2c_enabled) * AA_CONFIG_GPIO_I2C
        )


class PyAA_Probe:
    def __init__(self, port: int):
        self.port       = port
        self.handle     = None

        self.version    = None
        self.features   = None

        self.opened     = Event()

        # Drivers
        self.i2c_driver = None

        # Worker thread and locks
        self.w_lock     = RLock()
        self.r_lock     = RLock()

    # ───────────── Open / Close ───────────── #

    def open(self):
        if not self.opened.is_set():
            self.handle, infos_ext = aa_open_ext(self.port)
            
            if self.handle < 0:
                raise PyAA_Probe_Error(self.handle, "Unable to open device")

            self.opened.set()

            # Parse infos_ext
            self.features = PyAA_Features.from_byte(infos_ext.features)


    def close(self):
        if self.opened.is_set():
            aa_close(self.handle)
            self.opened.clear()


    # ──────────── Config get/set ──────────── #

    def config_get(self):
        with self.r_lock:
            return PyAA_Config.from_byte(aa_configure(self.handle, AA_CONFIG_QUERY))


    def config_set(self, conf: PyAA_Config):
        with self.w_lock:
            ret = aa_configure(self.handle, conf.byte())
            if ret < 0: raise PyAA_Probe_Error(ret, "Unable to set config")

    # ────── Target power enable/disable ───── #
    
    def target_power_enable(self, v: bool):
        with self.w_lock:
            ret = aa_target_power(self.handle, AA_TARGET_POWER_BOTH if v else AA_TARGET_POWER_NONE)
            if ret < 0: raise PyAA_Probe_Error(ret, "Unable to set Target power")


    # ───────── Context manager stuff ──────── #

    def __enter__(self):
        self.open()
        return self


    def __exit__(self, type, value, traceback):
        self.close()

# ┌────────────────────────────────────────┐
# │ Master I2C driver                      │
# └────────────────────────────────────────┘

class PyAA_I2C_Master_Driver:
    def __init__(self,
        probe: PyAA_Probe,
        pullups_enabled: bool = None,
        bitrate_khz: int      = None,
        timeout_ms: int       = 200
    ):
        """
        When values are none, the current config
        is untouched.
        """

        self.probe           = probe
        self.pullups_enabled = pullups_enabled
        self.bitrate_khz     = bitrate_khz
        self.timeout_ms      = timeout_ms

    # ─────────── Attach / Dettach ─────────── #
    
    def attach(self):
        """
        Attach this driver to the probe

        Example:
            with PyAA_Probe(port_number) as probe:
                with PyAA_I2C_Master_Driver(probe, pullups_enabled = True, bitrate_khz = 100) as i2c:
                    # Do stuff

                with PyAA_I2C_Slave_Driver(probe, ...) as i2c:
                    # Do other stuff
                
        """

        # Ensure that the probe is opened
        if not self.probe.opened.is_set():
            raise RuntimeError("Aardvark probe must be opened prior to attaching the driver")

        # Check features
        if not self.probe.features.i2c:
            raise RuntimeError("I2C is not supported by probe")

        # Attach the driver to the probe
        if self.probe.i2c_driver is not None:
            raise RuntimeError("An i2c driver is already attached to the probe")
        self.probe.i2c_driver = self

        # Update config
        conf = self.probe.config_get()
        conf.i2c_enabled = True
        self.probe.config_set(conf)

        # Set I2C as master, and set parameters
        with self.probe.w_lock:
            ret = aa_i2c_slave_disable(self.probe.handle)#
            if ret < 0: raise PyAA_Probe_Error(ret, "Cannot set I2C probe as Master")

            if self.bitrate_khz is not None:
                ret = aa_i2c_bitrate(self.probe.handle, self.bitrate_khz)
                if ret < 0: raise PyAA_Probe_Error(ret, "Cannot set I2C bitrate")

            if self.pullups_enabled is not None:
                ret = aa_i2c_pullup(self.probe.handle, AA_I2C_PULLUP_BOTH if self.pullups_enabled else AA_I2C_PULLUP_NONE)
                if ret < 0: raise PyAA_Probe_Error(ret, "Cannot set I2C pullups")

            if self.timeout_ms is not None:
                ret = aa_i2c_bus_timeout(self.probe.handle, self.timeout_ms)
                if ret < 0: raise PyAA_Probe_Error(ret, "Cannot set I2C bus timeout")

    def dettach(self):
        self.probe.i2c_driver = None

    # ───────────── Write / Read ───────────── #

    def write(self, slave_addr: int, data: bytes):
        with self.probe.w_lock:
            ret, num_written = aa_i2c_write_ext(self.probe.handle, slave_addr, 0, array("B", data))
            if ret < 0:          raise PyAA_Probe_Error(ret, "Unable to write I2C master data")
            if num_written == 0: raise RuntimeError("Written 0 bytes")

        return num_written

    def read(self, slave_addr: int):
        with self.probe.r_lock:
            ret, data_in, num_read = aa_i2c_read_ext(self.probe.handle, slave_addr, 0, MAX_I2C_READ_SIZE)

            if ret < 0: raise PyAA_Probe_Error(ret, "Unable to read I2C master data for slave 0x{slave_addr:02X}")
            #if num_read == 0: raise RuntimeError("Read 0 bytes")
        
        return bytes(data_in)

    # ──────────── Context manager ─────────── #
    
    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, type, value, traceback):
        self.dettach()


# ┌────────────────────────────────────────┐
# │ Slave I2C driver                       │
# └────────────────────────────────────────┘

class PyAA_I2C_Slave_Driver:
    def __init__(self,
        probe: PyAA_Probe,
        addr:  int,
        pullups_enabled: bool = None,
        timeout_ms: int = 200
    ):
        self.probe           = probe
        self.addr            = addr
        self.pullups_enabled = pullups_enabled
        self.timeout_ms      = timeout_ms

    # ─────────── Attach / Dettach ─────────── #
    
    def attach(self):
        # Ensure that the probe is opened
        if not self.probe.opened.is_set():
            raise RuntimeError("Aardvark probe must be opened prior to attaching the driver")

        # Check features
        if not self.probe.features.i2c:
            raise RuntimeError("I2C is not supported by probe")

        # Attach the driver to the probe
        if self.probe.i2c_driver is not None:
            raise RuntimeError("An i2c driver is already attached to the probe")
        self.probe.i2c_driver = self

        # Update config
        conf = self.probe.config_get()
        conf.i2c_enabled = True
        self.probe.config_set(conf)

        # Set I2C as slave, and set parameters
        with self.probe.w_lock:
            ret = aa_i2c_slave_enable(self.probe.handle, self.addr | 0x80, 0,0)
            if ret < 0: raise PyAA_Probe_Error(ret, "Unable to set probe as I2C slave")

            if self.pullups_enabled is not None:
                ret = aa_i2c_pullup(self.probe.handle, AA_I2C_PULLUP_BOTH if self.pullups_enabled else AA_I2C_PULLUP_NONE)
                if ret < 0: raise PyAA_Probe_Error(ret, "Cannot set I2C pullups")

            if self.timeout_ms is not None:
                ret = aa_i2c_bus_timeout(self.probe.handle, self.timeout_ms)
                if ret < 0: raise PyAA_Probe_Error(ret, "Cannot set I2C bus timeout")

    def dettach(self):
        self.probe.i2c_driver = None


    # ───────────── Write / Read  ──────────── #

    def read(self, timeout_ms=None):
        # Poll for read event
        # FIXME # Dirty skip of unwanted values, may breaks timeout stuff. Better event loop handling required!
        status = None
        while status != AA_ASYNC_I2C_READ:
            status = aa_async_poll(self.probe.handle, timeout_ms or -1)
            
            if status == AA_ASYNC_NO_DATA:
                raise TimeoutError("Timeout waiting for I2C data")

        # Read data
        ret, addr, data_in, num_read = aa_i2c_slave_read_ext(self.probe.handle, MAX_I2C_READ_SIZE)
        if ret < 0: raise PyAA_Probe_Error(ret, "Cannot read from I2C slave")
        #if num_read == 0: raise RuntimeError("Read 0 bytes")

        return (addr, bytes(data_in))


    def response_set(self, resp: bytes):
        ret = aa_i2c_slave_set_response(self.probe.handle, array('B', resp))
        if ret < 0: raise PyAA_Probe_Error(ret, "Cannot set I2C slave response")

    def write_wait(self, timeout_ms=None):
        """
        May disappear in the future, quick workaround!
        """
        # Poll for write event
        # FIXME # Dirty skip of unwanted values, may breaks timeout stuff. Better event loop handling required!
        status = None
        while status != AA_ASYNC_I2C_WRITE:
            status = aa_async_poll(self.probe.handle, timeout_ms or -1)
            
            if status == AA_ASYNC_NO_DATA:
                raise TimeoutError("Timeout waiting for slave I2C write")

        status, num_written = aa_i2c_write_stats_ext(self.probe.handle)
        if status < 0: raise PyAA_Probe_Error(status, "Failed slave I2C write")
        return num_written


    # ───────── Context manager stuff ──────── #
    
    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, type, value, traceback):
        self.dettach()
