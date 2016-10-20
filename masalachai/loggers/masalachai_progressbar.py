# -*- coding: utf-8 -*-

import progressbar

class MasalachaiProgressBar(progressbar.ProgressBar):
    def __init__(self, min_value=0, max_value=None, widgets=None,
            left_justify=True, initial_value=0, poll_interval=None,
            widget_kwargs=None, dynamic_data=[],
            **kwargs):
        super(MasalachaiProgressBar, self).__init__(min_value=min_value, max_value=max_value, widgets=widgets,
                left_justify=left_justify, initial_value=initial_value, poll_interval=poll_interval,
                widget_kwargs=widget_kwargs, **kwargs)
        self.dynamic_data = dynamic_data

    def data(self):
        d = super(MasalachaiProgressBar, self).data()
        d.update(self.dynamic_data)
        return d

    def update(self, value=None, force=False, **kwargs):
        'Updates the ProgressBar to a new value.'
        if self.start_time is None:
            self.start()
            return self.update(value)

        # Save the updated values for dynamic messages
        for key in kwargs:
            if key in self.dynamic_data:
                self.dynamic_data[key] = kwargs[key]
            elif key in self.dynamic_messages:
                self.dynamic_messages[key] = kwargs[key]
            else:
                raise TypeError(
                    'update() got an unexpected keyword ' +
                    'argument \'{}\''.format(key))

        if value is not None and value is not progressbar.base.UnknownLength:
            if self.max_value is progressbar.base.UnknownLength:
                # Can't compare against unknown lengths so just update
                pass
            elif self.min_value <= value <= self.max_value:
                # Correct value, let's accept
                pass
            else:
                raise ValueError(
                    'Value out of range, should be between %s and %s'
                    % (self.min_value, self.max_value))

            self.previous_value = self.value
            self.value = value

        if self._needs_update() or force:
            self.updates += 1
            progressbar.bar.ResizableMixin.update(self, value=value)
            progressbar.bar.ProgressBarBase.update(self, value=value)
            progressbar.bar.StdRedirectMixin.update(self, value=value)

