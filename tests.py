import os

from physical_models.models_two_state_obstructed_diffusion import TwoStateConfinedDiffusion
import numpy as np

from tracks.simulated_tracks import SimulatedTrack

if __name__ == '__main__':
    phys_model = TwoStateConfinedDiffusion.create_random()
    x_noisy, y_noisy, x, y, t, state, switching = phys_model.simulate_track(track_length=15, track_time=0.35)
    axes_data = np.zeros(shape=(2, 15))
    axes_data[0] = x_noisy
    axes_data[1] = y_noisy
    sim_track = SimulatedTrack(track_length=15, track_time=0.35, model_type=type(phys_model).__name__,
                               n_axes=2)
    sim_track.set_axes_data(axes_data)
    sim_track.set_time_axis(time_axis_data=t)

    sim_track.plot_xy()
    # End
