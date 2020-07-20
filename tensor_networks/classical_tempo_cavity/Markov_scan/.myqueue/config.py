config = {
    'scheduler': 'slurm',
    'parallel_python': 'gpaw python',
    'nodes': [
        ('xeon40', {'queue': 'slurm',
                    'cores': 40,
                    'memory': '380G'}),
        ('xeon24', {'queue': 'slurm',
                    'cores': 24,
                    'memory': '250G'}),
        ('xeon16', {'queue': 'slurm',
                    'cores': 16,
                    'memory': '60G'}),
        ('xeon8', {'queue': 'slurm',
                   'cores': 8,
                   'memory': '22G'}),
        ('xeon24_512', {'queue': 'slurm',
                        'cores': 24,
                        'memory': '500G'}),
        ('xeon40_768', {'queue': 'slurm',
                        'cores': 40,
                        'memory': '760G'})]}

