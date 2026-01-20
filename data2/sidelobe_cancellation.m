function y = sidelobe_cancellation(y_in, reference_chip, main_idx, guard_half_width, K)
%SIDELOBE_CANCELLATION  Template-based sidelobe cancellation in range domain
%   y = sidelobe_cancellation(y_in, reference_chip, main_idx, guard_half_width, K)
%
%   [Fix] Removed reference_chip truncation to ensure template 'r' matches
%   the length of input 'y_in', preventing dimension mismatch errors.

    if nargin < 3 || isempty(main_idx)
        [~, main_idx] = max(abs(y_in));
    end
    if nargin < 4 || isempty(guard_half_width)
        guard_half_width = round(length(y_in) * 0.02);
    end
    if nargin < 5 || isempty(K)
        K = 3; % cancel top-3 interference peaks
    end

    y_in = y_in(:).';
    N = length(y_in);

    % Autocorrelation template (range response)
    % MODIFIED: Use full length reference_chip to match y_in dimensions
    % 不要在这里截断 chip，否则会导致卷积后的 r 变短，后续 y - tmp 就会报错
    chip = reference_chip(:).';
    
    % Compute autocorrelation template
    % 'same' returns the central part of the convolution, same size as 'chip' (which is N)
    r = conv(chip, conj(fliplr(chip)), 'same');
    
    % Normalize template peak to 1
    [~, r0] = max(abs(r));
    r = r / (r(r0) + eps);

    y = y_in;

    % Protect target mainlobe region
    protect = false(1, N);
    L = max(1, main_idx - guard_half_width);
    R = min(N, main_idx + guard_half_width);
    protect(L:R) = true;

    % Find candidate peaks outside protected region
    mag = abs(y_in);
    mag(protect) = 0;

    % Simple peak picking: take top-K sample indices
    [~, sorted_idx] = sort(mag, 'descend');
    peak_idx = sorted_idx(1:min(K, numel(sorted_idx)));
    peak_idx = peak_idx(mag(peak_idx) > 0);

    for ii = 1:numel(peak_idx)
        k = peak_idx(ii);
        a = y_in(k); % complex amplitude at peak

        % Build shifted template centered at k
        % r is centered at r0; shift amount = k - r0
        shift = k - r0;
        
        % Circular shift is used to handle edge cases in periodic/windowed simulation
        r_shift = circshift(r, [0, shift]);

        % Subtract scaled template
        % Optional: local protection around the interference peak itself can be added if needed,
        % but standard SLC subtracts the estimate.
        
        % Local guard to avoid nuking the peak center (from original code logic)
        local_guard = round(guard_half_width * 0.5);
        kL = max(1, k - local_guard);
        kR = min(N, k + local_guard);

        tmp = a * r_shift;
        % The original code had this mask. Keep it to preserve logic intent.
        tmp(kL:kR) = 0; 

        % Subtraction (Now dimensions match: 1xN - 1xN)
        y = y - tmp;
    end
end