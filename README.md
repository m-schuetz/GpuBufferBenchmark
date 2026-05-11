
# GPU Buffer Benchmarks

Measuring performance of various types of buffers in CUDA kernels.

- Vanilla CUDA Buffers allocated via cuMemAlloc.
- Buffers allocated via CUDA Virtual Memory Management (VMM).
- VMM buffers with buffer compression enabled
- Buffers allocated in Vulkan, and imported into CUDA.

Under different conditions:

- Empty buffers. Just zeroes. 
- Buffers with different values that don't utilize the full bit range. (E.g. 18 bit integer in an uint32 array)
- Completely random u32 values. 



# Benchmark Results

Tested on an RTX 4090 with a memory bandwidth of 1TB/s.

Test Data:
<table>
	<tr>
		<th></th>
		<th></th>
	</tr><tr>
		<th>Constant Value Array</th>
		<td>500 million uint32 values, all of them with the value "123456"</td>
	</tr><tr>
		<th>Simple Value Array</th>
		<td>500 million uint32 values. Repeats the values (0, 1, 2, ..., 262'144) throughout the buffer. </td>
	</tr><tr>
		<th>Random Value Array</th>
		<td>500 million random uint32 values.</td>
	</tr>
</table>

We first benchmark aligned access, followed by access to "18-bit integers". The latter simply reinterprets the test data as 18-bit buffers, 

<h2>Aligned U32</h2>

Benchmark access to a buffer with 500 million uint32_t values.

<table>
  <thead>
    <tr>
      <th rowspan="2">Label</th>
      <th colspan="2">Constant Value Array</th>
      <th colspan="2">Simple Value Array</th>
      <th colspan="2">Random Value Array</th>
    </tr>
    <tr>
      <th>Duration</th>
      <th>GB/s</th>
      <th>Duration</th>
      <th>GB/s</th>
      <th>Duration</th>
      <th>GB/s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>BASELINE</td>
      <td align="right">2.096</td>
      <td align="right">954.1</td>
      <td align="right">2.094</td>
      <td align="right">955.1</td>
      <td align="right">2.094</td>
      <td align="right">955.1</td>
    </tr>
    <tr>
      <td>Virtual Memory</td>
      <td align="right">2.097</td>
      <td align="right">953.7</td>
      <td align="right">2.093</td>
      <td align="right">955.5</td>
      <td align="right">2.095</td>
      <td align="right">954.6</td>
    </tr>
    <tr>
      <td>Virtual Memory (compressed)</td>
      <td align="right">1.265</td>
      <td align="right">1581.4</td>
      <td align="right">1.279</td>
      <td align="right">1563.8</td>
      <td align="right">2.175</td>
      <td align="right">919.6</td>
    </tr>
    <tr>
      <td>Vulkan Imported</td>
      <td align="right">1.260</td>
      <td align="right">1587.9</td>
      <td align="right">1.282</td>
      <td align="right">1560.0</td>
      <td align="right">2.174</td>
      <td align="right">920.0</td>
    </tr>
  </tbody>
</table>

<h2>Non-Aligned (18 bit)</h2>

Benchmark access to a buffer with 500 million 18-bit integers. 

<table>
  <thead>
    <tr>
      <th rowspan="2">Label</th>
      <th colspan="2">Constant Value Array</th>
      <th colspan="2">Simple Value Array</th>
      <th colspan="2">Random Value Array</th>
    </tr>
    <tr>
      <th>Duration</th>
      <th>GB/s</th>
      <th>Duration</th>
      <th>GB/s</th>
      <th>Duration</th>
      <th>GB/s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>BASELINE</td>
      <td align="right">1.420</td>
      <td align="right">792.1</td>
      <td align="right">1.423</td>
      <td align="right">790.4</td>
      <td align="right">1.421</td>
      <td align="right">791.5</td>
    </tr>
    <tr>
      <td>Virtual Memory</td>
      <td align="right">1.421</td>
      <td align="right">791.5</td>
      <td align="right">1.424</td>
      <td align="right">789.8</td>
      <td align="right">1.422</td>
      <td align="right">791.0</td>
    </tr>
    <tr>
      <td>Virtual Memory (compressed)</td>
      <td align="right">1.391</td>
      <td align="right">808.5</td>
      <td align="right">1.409</td>
      <td align="right">798.5</td>
      <td align="right">1.462</td>
      <td align="right">769.4</td>
    </tr>
    <tr>
      <td>Vulkan Imported</td>
      <td align="right">1.389</td>
      <td align="right">810.2</td>
      <td align="right">1.409</td>
      <td align="right">798.4</td>
      <td align="right">1.463</td>
      <td align="right">769.0</td>
    </tr>
  </tbody>
</table>