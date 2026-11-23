import { useEffect, useRef, useState } from 'react';

export interface SelectOption { value: string; label: string }
interface SelectProps { value: string; options: SelectOption[]; onChange: (value: string) => void; disabled?: boolean; ariaLabel?: string }

export default function Select({ value, options, onChange, disabled = false, ariaLabel }: SelectProps) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const selected = options.find(option => option.value === value) || options[0];

  useEffect(() => {
    const close = (event: MouseEvent) => { if (root.current && !root.current.contains(event.target as Node)) setOpen(false); };
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', close); document.addEventListener('keydown', escape);
    return () => { document.removeEventListener('mousedown', close); document.removeEventListener('keydown', escape); };
  }, []);

  return <div className={`select-root${open ? ' is-open' : ''}`} ref={root}>
    <button type="button" className="select-trigger" aria-haspopup="listbox" aria-expanded={open} aria-label={ariaLabel} disabled={disabled} onClick={() => setOpen(current => !current)}>
      <span>{selected?.label}</span><span className="select-chevron" aria-hidden="true">⌄</span>
    </button>
    {open && <div className="select-menu" role="listbox" aria-label={ariaLabel}>
      {options.map(option => <button type="button" role="option" aria-selected={option.value === value} className={`select-option${option.value === value ? ' is-selected' : ''}`} key={option.value} onClick={() => { onChange(option.value); setOpen(false); }}><span>{option.label}</span>{option.value === value && <span className="select-check" aria-hidden="true">✓</span>}</button>)}
    </div>}
  </div>;
}
