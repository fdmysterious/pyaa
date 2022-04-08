==================================
PyAA: Python Aaardvark API wrapper
==================================

:Authors:   - Florian Dupeyron <florian.dupeyron@mugcat.fr>
:Date:     April 2022

This is a simple wrapper around the `aardvark_py`_ binding of the closed-source
C aardvark API library.

.. _`aardvark_py`: https://github.com/totalphase/aardvark_py

Limitations
===========

- Concurrent use of I2C slave and SPI not supported (yet)
- Probably **not** thread safe
- Missing SPI support (yet)
- Missing GPIO support (yet)

Example usage: i2c master
=========================

.. code:: python

    from pyaa import (
        list_probes,

        PyAA_Probe,
        PyAA_I2C_Master_Driver,
        PyAA_I2C_Slave_Driver
    )

    # Indicative values for the example
    SLAVE_ADDR     = 0x40
    SLAVE_REG_ADDR = 0x25

    if __name__ == "__main__":
        # Find first available probe
        probelist = list_probes()

        def first_available_probe(probes):
            return next(filter(lambda x: x.free), probes)

        probe_info = first_available_probe(probelist)

        # Open probe
        with PyAA_Probe(probe_info.port_number) as probe:
            # Open I2C master driver
            with PyAA_I2C_Master_Driver(probe, pullups_enabled=True, bitrate_khz=100) as i2c:
                req_data = bytes([SLAVE_REG_ADDR])

                # TODO # Replace with i2c.write_read()
                # Send register read request
                i2c.write(slave_addr, req_data)

                # Read device response
                data = i2c.read(slave_addr)

                print(f"Read data: {data}")
