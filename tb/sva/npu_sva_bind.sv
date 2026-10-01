// Binds the assertion/coverage module into every npu_top instance; ports are
// connected by name to npu_top's internal signals
bind npu_top npu_sva #(.ADDR_W(ADDR_W), .MEM_DW(MEM_DW)) u_sva (.*);
