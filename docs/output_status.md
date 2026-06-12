# Output Status

`output_status.csv` explains every expected artifact.

Statuses include `active`, `empty`, `skipped_disabled`, `skipped_missing_dependency`,
`skipped_no_data`, `skipped_provider_lacks_capability`, and `error`.

A CSV existing in a directory does not mean the module was active. Trust `output_status.csv` and
`active_modules.csv` for module interpretation.
