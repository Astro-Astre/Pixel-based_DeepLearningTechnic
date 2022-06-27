merger = df_auto.query('merging_minor_disturbance_fraction > %f '
                               '| merging_major_disturbance_fraction > %f '
                               '| merging_merger_fraction > %f '
                               % (0.6, 0.6, 0.6))
smoothRounded = df_auto.query('smooth_or_featured_smooth_fraction >  %f '
                       '& how_rounded_round_fraction > %f' % (0.7, 0.8))
smoothInBetween = df_auto.query('smooth_or_featured_smooth_fraction >  %f '
                       '& how_rounded_in_between_fraction > %f' % (0.7, 0.85))
smoothCigarShaped = df_auto.query('smooth_or_featured_smooth_fraction > %f '
                                  '& how_rounded_cigar_shaped_fraction > %f' % (0.5, 0.6))
edgeOn = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
                            '& disk_edge_on_yes_fraction > %f'
                            % (0.5, 0.7))
diskNoBar = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
                                '& disk_edge_on_no_fraction > %f '
                                '& bar_no_fraction > %f '
                                % (0.5, 0.5, 0.7))
diskStrongBar = df_auto.query('smooth_or_featured_featured_or_disk_fraction > %f '
                                '& disk_edge_on_no_fraction > %f '
                                '&bar_strong_fraction > %f '
                                % (0.5, 0.5, 0.6))