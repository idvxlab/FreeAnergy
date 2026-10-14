export interface ExecResult {
	output: string;
	exitCode: number;
	/**
	 * True when the failure is infrastructure-domain (container not running /
	 * no such container / sidecar timeout), not a task-domain command failure.
	 */
	infra?: boolean;
}
