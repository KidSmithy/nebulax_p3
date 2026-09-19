import { SubmissionCsv } from '../types/telemetry';

const cell = (v: string | number) => {
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

export function submissionToCsv({ columns, rows }: SubmissionCsv): string {
  return [columns, ...rows].map((row) => row.map(cell).join(',')).join('\n') + '\n';
}

/** Saves the submission as `<filename>` in the browser's download folder. */
export function downloadSubmission(submission: SubmissionCsv): void {
  const blob = new Blob([submissionToCsv(submission)], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = submission.filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
