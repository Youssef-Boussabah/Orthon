"""Shared measurement options."""
from .timing import TimingConfig


def add_options(parser, *, timing=True):
    parser.add_argument('--quick', action='store_true', help='bounded smoke; representative=false')
    parser.add_argument('--json', metavar='PATH', help='strict orthon-benchmark v1 result')
    if timing:
        parser.add_argument('--warmup', type=int, default=25)
        parser.add_argument('--samples', type=int, default=11)
        parser.add_argument('--target-ms', type=float, default=40.)
        parser.add_argument('--max-sample-ms', type=float, default=500.)
        parser.add_argument('--inner', type=int)
    return parser


def timing_config(args):
    if args.quick:
        return TimingConfig(warmup=1,samples=3,fixed_inner=1)
    return TimingConfig(warmup=args.warmup,samples=args.samples,target_ms=args.target_ms,
                        max_sample_ms=args.max_sample_ms,fixed_inner=args.inner)
