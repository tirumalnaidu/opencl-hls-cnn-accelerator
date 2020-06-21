#ifndef AOCL_MMD_H
#define AOCL_MMD_H

/* (C) 1992-2014 Altera Corporation. All rights reserved.                          */
/* Your use of Altera Corporation's design tools, logic functions and other        */
/* software and tools, and its AMPP partner logic functions, and any output        */
/* files any of the foregoing (including device programming or simulation          */
/* files), and any associated documentation or information are expressly subject   */
/* to the terms and conditions of the Altera Program License Subscription          */
/* Agreement, Altera MegaCore Function License Agreement, or other applicable      */
/* license agreement, including, without limitation, that your use is for the      */
/* sole purpose of programming logic devices manufactured by Altera and sold by    */
/* Altera or its authorized distributors.  Please refer to the applicable          */
/* agreement for further details.                                                  */
    


#ifdef __cplusplus
extern "C" {
#endif

/* Support for memory mapped ACL devices.
 *
 * Typical API lifecycle, from the perspective of the caller.
 *
 *    1. aocl_mmd_open must be called first, to provide a handle for further
 *    operations.
 *
 *    2. The interrupt and status handlers must be set.
 *   
 *    3. Read and write operations are performed.
 *
 *    4. aocl_mmd_close may be called to shut down the device.  No further
 *    operations are permitted until a subsequent aocl_mmd_open call.
 *
 * aocl_mmd_get_offline_info can be called anytime including before
 * open. aocl_mmd_get_info can be called anytime between open and close.
 */

#ifndef AOCL_MMD_CALL
#if defined(_WIN32)
#define AOCL_MMD_CALL   __declspec(dllimport)
#else
#define AOCL_MMD_CALL
#endif
#endif

/* The MMD API's version - the runtime expects this string when
 * AOCL_MMD_VERSION is queried.  This changes only if the API has changed */
#define AOCL_MMD_VERSION_STRING "14.0"

typedef void* aocl_mmd_op_t;

typedef struct {
   unsigned lo; /* 32 least significant bits of time value. */
   unsigned hi; /* 32 most significant bits of time value. */
} aocl_mmd_timestamp_t;


/* Defines the set of characteristics that can be probed about the board before
 * opening a device.  The type of data returned be each is specified in
 * parentheses in the adjacent comment.
 *
 * AOCL_MMD_NUM_BOARDS and AOCL_MMD_BOARD_NAMES
 *   These two fields can be used to implement multi-device support.  The MMD
 *   layer may have a list of devices it is capable of interacting with, each
 *   identified with a unique name.  The length of the list should be returned
 *   in AOCL_MMD_NUM_BOARDS, and the names of these devices returned in
 *   AOCL_MMD_BOARD_NAMES.  The OpenCL runtime will try to call aocl_mmd_open 
 *   for each board name returned in AOCL_MMD_BOARD_NAMES.
 *
 * */
typedef enum {
   AOCL_MMD_VERSION = 0,       /* Version of MMD (char*)*/
   AOCL_MMD_NUM_BOARDS = 1,    /* Number of candidate boards (int)*/
   AOCL_MMD_BOARD_NAMES = 2,   /* Names of boards available delimiter=; (char*)*/
   AOCL_MMD_VENDOR_NAME = 3,   /* Name of vendor (char*) */
   AOCL_MMD_VENDOR_ID = 4,     /* An integer ID for the vendor (int) */
} aocl_mmd_offline_info_t;

/* Defines the set of characteristics that can be probed about the board after
 * opening a device.  This can involve communication to the device 
 *
 * AOCL_MMD_NUM_KERNEL_INTERFACES - The number of kernel interfaces, usually 1
 *
 * AOCL_MMD_KERNEL_INTERFACES - the handle for each kernel interface.  
 *      param_value will have size AOCL_MMD_NUM_KERNEL_INTERFACES * sizeof int
 *
 * AOCL_MMD_PLL_INTERFACES - the handle for each pll associated with each
 * kernel interface.  If a kernel interface is not clocked by acl_kernel_clk
 * then return -1
 *
 * */
typedef enum {
   AOCL_MMD_NUM_KERNEL_INTERFACES = 1,  /* Number of Kernel interfaces (int) */
   AOCL_MMD_KERNEL_INTERFACES = 2,      /* Kernel interface (int*) */
   AOCL_MMD_PLL_INTERFACES = 3,         /* Kernel clk handles (int*) */
   AOCL_MMD_MEMORY_INTERFACE = 4,       /* Global memory handle (int) */
   AOCL_MMD_TEMPERATURE = 5,            /* Temperature measurement (float) */
   AOCL_MMD_PCIE_INFO = 6,              /* PCIe information (char*) */
   AOCL_MMD_BOARD_NAME = 7,             /* Name of board (char*) */
   AOCL_MMD_BOARD_UNIQUE_ID = 8,        /* Unique ID of board (int) */
   AOCL_MMD_POWER = 9,                  /* Power usage of board (Watts) (float) */
   AOCL_MMD_NALLA_DIAGNOSTIC = 10,      /* Board specific diagnostics. */
   AOCL_MMD_MAC0_ADDRESS,
   AOCL_MMD_MAC1_ADDRESS,
   AOCL_MMD_MAC2_ADDRESS,
   AOCL_MMD_MAC3_ADDRESS

} aocl_mmd_info_t;

typedef void (*aocl_mmd_interrupt_handler_fn)( int handle, void* user_data );
typedef void (*aocl_mmd_status_handler_fn)( int handle, void* user_data, aocl_mmd_op_t op, int status );


/* Get information about the board using the enum aocl_mmd_offline_info_t for
 * offline info (called without a handle), and the enum aocl_mmd_info_t for
 * info specific to a certain board.  
 * Arguments:
 *
 *   requested_info_id - a value from the aocl_mmd_offline_info_t enum
 *
 *   param_value_size - size of the param_value field in bytes.  This should
 *     match the size of the return type expected as indicated in the enum
 *     definition.  For example, the AOCL_MMD_TEMPERATURE returns a float, so
 *     the param_value_size should be set to sizeof(float) and you should
 *     expect the same number of bytes returned in param_size_ret.
 *
 *   param_value - pointer to the variable that will receive the returned info
 *
 *   param_size_ret - receives the number of bytes of data actually returned
 *
 * Returns: a negative value to indicate error.
 */
AOCL_MMD_CALL int aocl_mmd_get_offline_info(
    aocl_mmd_offline_info_t requested_info_id,
    size_t param_value_size,
    void* param_value,
    size_t* param_size_ret );

AOCL_MMD_CALL int aocl_mmd_get_info(
    int handle,
    aocl_mmd_info_t requested_info_id,
    size_t param_value_size,
    void* param_value,
    size_t* param_size_ret );

AOCL_MMD_CALL int aocl_mmd_card_info(
    const char * device_name,
    aocl_mmd_info_t requested_info_id,
    size_t param_value_size,
    void* param_value,
    size_t* param_size_ret );	
	
/* Open and initialize the named device.  
 *
 * The name is typically one specified by the AOCL_MMD_BOARD_NAMES offline
 * info.
 * 
 * Arguments:
 *    name - open the board with this name (provided as a C-style string, 
 *           i.e. NUL terminated ASCII.)
 *
 * Returns: the non-negative integer handle for the board, otherwise a
 * negative value to indicate error.  Upon receiving the error, the OpenCL
 * runtime will proceed to open other known devices, hence the MMD mustn't
 * exit the application if an open call fails.
 */
AOCL_MMD_CALL int aocl_mmd_open(const char *name); 

/* Close an opened device, by its handle.
 * Returns: 0 on success, negative values on error. 
 */
AOCL_MMD_CALL int aocl_mmd_close(int handle);

/* Set the interrupt handler for the opened device.
 * The interrupt handler is called whenever the client needs to be notified
 * of an asynchronous event signalled by the device internals.
 * For example, the kernel is stalled has completed or is stalled.
 *
 * Important: Interrupts from the kernel must be ignored until this handler is
 * set
 *
 * Arguments:
 *   fn - the callback function to invoke when a kernel interrupt occurs
 *   user_data - the data that should be passed to fn when it is called.
 *
 * Returns: 0 if successful, negative on error
 */
AOCL_MMD_CALL int aocl_mmd_set_interrupt_handler( int handle, aocl_mmd_interrupt_handler_fn fn, void* user_data );

/* Set the operation status handler for the opened device.
 * The operation status handler is called with
 *    status 0 when the operation has completed successfully.
 *    status negative when the operation completed with errors.
 *
 * Arguments:
 *   fn - the callback function to invoke when a status update is to be
 *   performed.
 *   user_data - the data that should be passed to fn when it is called.
 *
 * Returns: 0 if successful, negative on error
 */
AOCL_MMD_CALL int aocl_mmd_set_status_handler( int handle, aocl_mmd_status_handler_fn fn, void* user_data );

/* Called when the host is idle and hence possibly waiting for events to be
 * processed by the device
 *
 *
 * ** IMPORTANT **
 *
 * Returns: non-zero if the yield function performed useful work such as processing DMA transactions
 * Returns: 0 if there is no useful work to be performed
 *
 * NOTE: yield may be called continuously as long as it reports that it has useful work
 */
AOCL_MMD_CALL int aocl_mmd_yield(int handle);

/* Read, write and copy operations on a single interface.
 * If op is NULL
 *    - Then these calls must block until the operation is complete.
 *    - The status handler is not called for this operation.
 *
 * If op is non-NULL, then:
 *    - These may be non-blocking calls
 *    - The status handler must be called upon completion, with status 0
 *    for success, and a negative value for failure.
 *
 * Arguments:
 *   op - the operation object used to track this operations progress
 *
 *   len - the size in bytes to transfer
 *
 *   src - the host buffer being read from
 *
 *   dst - the host buffer being written to
 *
 *   inteface - the handle to the interface being accessed.  E.g. To
 *   access global memory this handle will be whatever is returned by
 *   aocl_mmd_get_info when called with AOCL_MMD_MEMORY_INTERFACE.
 *
 *   offset/src_offset/dst_offset - the byte offset within the interface that
 *   the transfer will begin at.
 *
 * The return value is 0 if the operation launch was successful, and
 * negative otherwise.
 */
AOCL_MMD_CALL int aocl_mmd_read(
      int handle,
      aocl_mmd_op_t op,
      size_t len,
      void* dst,
      int interface, size_t offset );
AOCL_MMD_CALL int aocl_mmd_write(
      int handle,
      aocl_mmd_op_t op,
      size_t len,
      const void* src,
      int interface, size_t offset );
AOCL_MMD_CALL int aocl_mmd_copy(
      int handle,
      aocl_mmd_op_t op,
      size_t len,
      int interface, size_t src_offset, size_t dst_offset );

/* Reprogram the device 
 *
 * The host will guarantee that no operations are currently executing on the
 * device.  That means the kernels will be idle and no read/write/copy
 * commands are active.  Interrupts should be disabled and the FPGA should
 * be reprogrammed with the data from user_data which has size size.  The host
 * will then call aocl_mmd_set_status_handler and aocl_mmd_set_interrupt_handler
 * again.  At this point interrupts can be enabled.
 *
 * The new handle to the board after reprogram does not have to be the same with
 * the one before.
 *
 * Arguments:
 *   user_data - The binary contents of the fpga.bin file created during
 *   Quartus compilation.
 *   size - the size in bytes of user_data
 *
 * Returns: the new non-negative integer handle for the board, otherwise a 
 * negative value to indicate error.
 */
AOCL_MMD_CALL int aocl_mmd_reprogram( int handle, void * user_data , size_t size);

/* Shared memory allocator
 * Allocates memory that is shared between the host and the FPGA.  The 
 * host will access this memory using the pointer returned by alloc, while the FPGA
 * will access the shared memory using device_ptr_out.  If shared memory is not
 * supported this should return NULL.
 *
 * Shared memory survives FPGA reprogramming if the CPU is not rebooted.
 *
 * Arguments:
 *   size - the size of the shared memory to allocate
 *   device_ptr_out - will receive the pointer value used by the FPGA (the device)
 *                    to access the shared memory.  Cannot be NULL.  The type is
 *                    unsigned long long to handle the case where the host has a 
 *                    smaller pointer size than the device.
 *
 * Returns: The pointer value to be used by the host to access the shared
 * memory if successful, otherwise NULL.
 */
AOCL_MMD_CALL void * aocl_mmd_shared_mem_alloc( int handle, size_t size, unsigned long long *device_ptr_out );

/* Shared memory de-allocator
 * Frees previously allocated shared memory.  If shared memory is not supported,
 * this function should do nothing.
 * 
 * Arguments: 
 *   host_ptr - the host pointer that points to the shared memory, as returned by 
 *              aocl_mmd_shared_mem_alloc
 *   size     - the size of the shared memory to free. Must 
 */
AOCL_MMD_CALL void aocl_mmd_shared_mem_free ( int handle, void* host_ptr, size_t size );

#ifdef __cplusplus
}
#endif

#endif
